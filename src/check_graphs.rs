//! Finite-size checks of the identities used in the Supplemental Material.
//!
//! GRAPH  28 pairwise non-isomorphic vertex-transitive graphs: Poisson equation, constant
//!        self-potential and norm, energy identities, hitting times, S^2 <= A / lambda_2, the
//!        speed limit for random local steps and, on the vertex- and edge-transitive
//!        families, the commutator identity and its triangle-free improvement.
//!        Triangle-supported steps test the on-site cross term when triangles are present.
//! FOLD   folded routing of the torus walk on the open lattice, L = 3, ..., 32.
//! LOG    bounds on A - (ln L) / (2 pi) on the square torus.
//! Any violated assertion aborts the program.
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(Clone, Copy, Debug, Default)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn r(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn abs(self) -> f64 {
        self.abs2().sqrt()
    }
}
impl Add for C {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
}
impl AddAssign for C {
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}
impl Sub for C {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
}
impl Mul<f64> for C {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self {
            re: self.re * b,
            im: self.im * b,
        }
    }
}
impl Mul for C {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
}
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / ((1_u64 << 53) as f64)
    }
    fn c(&mut self) -> C {
        C {
            re: 2.0 * self.unit() - 1.0,
            im: 2.0 * self.unit() - 1.0,
        }
    }
}
fn dot(a: &[C], b: &[C]) -> C {
    a.iter()
        .zip(b)
        .fold(C::default(), |s, (&x, &y)| s + x.conj() * y)
}
fn norm2(a: &[C]) -> f64 {
    a.iter().map(|z| z.abs2()).sum()
}
fn normalize(a: &mut [C]) {
    let c = norm2(a).sqrt().recip();
    for z in a {
        *z = *z * c;
    }
}
fn random_unitary(d: usize, rng: &mut Rng) -> Vec<C> {
    let mut cols: Vec<Vec<C>> = Vec::new();
    for _ in 0..d {
        let mut a: Vec<C> = (0..d).map(|_| rng.c()).collect();
        for _ in 0..2 {
            for q in &cols {
                let z = dot(q, &a);
                for j in 0..d {
                    a[j] = a[j] - q[j] * z;
                }
            }
        }
        normalize(&mut a);
        cols.push(a);
    }
    (0..d * d).map(|i| cols[i % d][i / d]).collect()
}

struct Graph {
    name: String,
    adj: Vec<Vec<usize>>,
    square: Option<usize>,
    // Known symmetry of the constructed family, not inferred from regularity.
    edge_transitive: bool,
}
impl Graph {
    fn new(
        name: String,
        n: usize,
        edges: Vec<(usize, usize)>,
        square: Option<usize>,
        edge_transitive: bool,
    ) -> Self {
        let mut adj = vec![Vec::new(); n];
        for (x, y) in edges {
            assert!(x != y);
            adj[x].push(y);
            adj[y].push(x);
        }
        for row in &mut adj {
            row.sort_unstable();
            row.dedup();
        }
        let d = adj[0].len();
        assert!(d > 0 && adj.iter().all(|r| r.len() == d));
        Self {
            name,
            adj,
            square,
            edge_transitive,
        }
    }
    fn n(&self) -> usize {
        self.adj.len()
    }
    fn triangle(&self) -> Option<[usize; 3]> {
        for x in 0..self.n() {
            for &y in self.adj[x].iter().filter(|&&y| y > x) {
                for &z in self.adj[y].iter().filter(|&&z| z > y) {
                    if self.adj[x].contains(&z) {
                        return Some([x, y, z]);
                    }
                }
            }
        }
        None
    }
    fn lap(&self) -> Vec<f64> {
        let n = self.n();
        let mut a = vec![0.0; n * n];
        for x in 0..n {
            a[x * n + x] = self.adj[x].len() as f64;
            for &y in &self.adj[x] {
                a[x * n + y] = -1.0;
            }
        }
        a
    }
}
fn cycle(n: usize) -> Graph {
    Graph::new(
        format!("cycle_{n}"),
        n,
        (0..n).map(|x| (x, (x + 1) % n)).collect(),
        None,
        true,
    )
}
fn complete(n: usize) -> Graph {
    Graph::new(
        format!("complete_{n}"),
        n,
        (0..n)
            .flat_map(|x| (x + 1..n).map(move |y| (x, y)))
            .collect(),
        None,
        true,
    )
}
fn torus(l: usize, h: usize) -> Graph {
    let mut e = Vec::new();
    for y in 0..h {
        for x in 0..l {
            e.push((y * l + x, y * l + (x + 1) % l));
            e.push((y * l + x, ((y + 1) % h) * l + x));
        }
    }
    Graph::new(
        format!("torus_{l}x{h}"),
        l * h,
        e,
        if l == h { Some(l) } else { None },
        l == h,
    )
}
fn cube(bits: usize) -> Graph {
    let n = 1 << bits;
    let mut e = Vec::new();
    for x in 0..n {
        for b in 0..bits {
            let y = x ^ (1 << b);
            if x < y {
                e.push((x, y));
            }
        }
    }
    Graph::new(format!("cube_{bits}"), n, e, None, true)
}
fn circulant(n: usize) -> Graph {
    Graph::new(
        format!("circulant_{n}_1_2"),
        n,
        (0..n)
            .flat_map(|x| [(x, (x + 1) % n), (x, (x + 2) % n)])
            .collect(),
        None,
        false,
    )
}
fn inverse(a: &[f64], n: usize) -> Vec<f64> {
    let w = 2 * n;
    let mut t = vec![0.0; n * w];
    for i in 0..n {
        for j in 0..n {
            t[i * w + j] = a[i * n + j];
        }
        t[i * w + n + i] = 1.0;
    }
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| t[i * w + k].abs().total_cmp(&t[j * w + k].abs()))
            .unwrap();
        assert!(t[pivot * w + k].abs() > 1e-12);
        for j in 0..w {
            t.swap(k * w + j, pivot * w + j);
        }
        let z = t[k * w + k];
        for j in 0..w {
            t[k * w + j] /= z;
        }
        for i in 0..n {
            if i != k {
                let z = t[i * w + k];
                for j in 0..w {
                    t[i * w + j] -= z * t[k * w + j];
                }
            }
        }
    }
    (0..n * n).map(|i| t[(i / n) * w + n + i % n]).collect()
}
fn eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    let mut a = a.to_vec();
    let mut converged = false;
    for _ in 0..100 * n * n {
        let (mut p, mut q, mut mx) = (0, 1, 0.0);
        for i in 0..n {
            for j in i + 1..n {
                if a[i * n + j].abs() > mx {
                    p = i;
                    q = j;
                    mx = a[i * n + j].abs();
                }
            }
        }
        if mx < 2e-13 {
            converged = true;
            break;
        }
        let v = a[p * n + q];
        let tau = (a[q * n + q] - a[p * n + p]) / (2.0 * v);
        let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
        let t = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = (1.0 + t * t).sqrt().recip();
        let s = t * c;
        a[p * n + p] -= t * v;
        a[q * n + q] += t * v;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
        for k in 0..n {
            if k != p && k != q {
                let v = a[k * n + p];
                let w = a[k * n + q];
                a[k * n + p] = c * v - s * w;
                a[p * n + k] = a[k * n + p];
                a[k * n + q] = s * v + c * w;
                a[q * n + k] = a[k * n + q];
            }
        }
    }
    assert!(converged);
    let mut e: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    e.sort_by(f64::total_cmp);
    e
}
type Sparse = Vec<Vec<(usize, C)>>;
fn apply(u: &Sparse, a: &[C]) -> Vec<C> {
    u.iter()
        .map(|r| r.iter().fold(C::default(), |s, &(j, z)| s + z * a[j]))
        .collect()
}
fn arc_control(g: &Graph, rng: &mut Rng) -> (Sparse, usize) {
    let n = g.n();
    let d = g.adj[0].len();
    let w = d + 1;
    let pre: Vec<Vec<C>> = (0..n).map(|_| random_unitary(w, rng)).collect();
    let post: Vec<Vec<C>> = (0..n).map(|_| random_unitary(w, rng)).collect();
    let mut u = vec![Vec::new(); n * w];
    for x in 0..n {
        for i in 0..w {
            for j in 0..w {
                let (y, k) = if j == d {
                    (x, d)
                } else {
                    let y = g.adj[x][j];
                    (y, g.adj[y].iter().position(|&v| v == x).unwrap())
                };
                for h in 0..w {
                    u[x * w + i].push((y * w + h, post[x][i * w + j] * pre[y][k * w + h]));
                }
            }
        }
    }
    (u, w)
}
fn plaquette_control(l: usize, w: usize, rng: &mut Rng) -> Sparse {
    let mut u: Sparse = (0..l * l * w).map(|j| vec![(j, C::r(1.0))]).collect();
    for y in (0..l - 1).step_by(2) {
        for x in (0..l - 1).step_by(2) {
            let v = [
                y * l + x,
                (y + 1) * l + x + 1,
                y * l + x + 1,
                (y + 1) * l + x,
            ];
            let a: Vec<usize> = v[..2]
                .iter()
                .flat_map(|&p| (0..w).map(move |j| p * w + j))
                .collect();
            let b: Vec<usize> = v[2..]
                .iter()
                .flat_map(|&p| (0..w).map(move |j| p * w + j))
                .collect();
            let q = 2 * w;
            let r = random_unitary(q, rng);
            let theta = 0.15 + 1.3 * rng.unit();
            let (s, c) = theta.sin_cos();
            for i in 0..q {
                u[a[i]] = vec![(a[i], C::r(c))];
                u[b[i]] = vec![(b[i], C::r(c))];
                for j in 0..q {
                    u[a[i]].push((b[j], r[i * q + j] * s));
                    u[b[i]].push((a[j], r[j * q + i].conj() * (-s)));
                }
            }
        }
    }
    u
}
fn triangle_step(n: usize, triangle: [usize; 3], block: &[C]) -> Sparse {
    let mut u: Sparse = (0..n).map(|i| vec![(i, C::r(1.0))]).collect();
    for i in 0..3 {
        u[triangle[i]] = (0..3).map(|j| (triangle[j], block[3 * i + j])).collect();
    }
    u
}
fn unitarity(u: &Sparse) -> f64 {
    let dim = u.len();
    let mut gram = vec![C::default(); dim * dim];
    for row in u {
        for &(i, z) in row {
            for &(j, w) in row {
                gram[i * dim + j] += z.conj() * w;
            }
        }
    }
    let mut mx: f64 = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            mx = mx.max((gram[i * dim + j] - C::r(if i == j { 1.0 } else { 0.0 })).abs());
        }
    }
    mx
}
fn commutator(u: &Sparse, w: usize, k: &[f64], n: usize, a: &[C]) -> f64 {
    let mut total = 0.0;
    for m in 0..n {
        for (i, row) in u.iter().enumerate() {
            let x = i / w;
            let mut z = C::default();
            for &(j, t) in row {
                z += t * a[j] * (k[x * n + m] - k[(j / w) * n + m]);
            }
            total += z.abs2();
        }
    }
    total / n as f64
}
fn onsite(u: &Sparse, w: usize, a: &[C]) -> Vec<C> {
    u.iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .filter(|&&(j, _)| i / w == j / w)
                .fold(C::default(), |s, &(j, z)| s + z * a[j])
        })
        .collect()
}
fn check_graph(g: &Graph, rng: &mut Rng) {
    let n = g.n();
    let d = g.adj[0].len();
    let triangle = g.triangle();
    let triangle_free = triangle.is_none();
    let lap = g.lap();
    let p = 1.0 / n as f64;
    let shifted: Vec<f64> = lap.iter().map(|v| v + p).collect();
    let k: Vec<f64> = inverse(&shifted, n).iter().map(|v| v - p).collect();
    let a = (0..n).map(|i| k[i * n + i]).sum::<f64>() / n as f64;
    let s2 = (0..n).map(|i| k[i * n] * k[i * n]).sum::<f64>();
    let mut residual: f64 = 0.0;
    let mut diagonal: f64 = 0.0;
    let mut normspan: f64 = 0.0;
    let mut energy: f64 = 0.0;
    let mut sigma: f64 = 0.0;
    let mut hitting: f64 = 0.0;
    for m in 0..n {
        diagonal = diagonal.max((k[m * n + m] - a).abs());
        normspan = normspan.max(((0..n).map(|x| k[x * n + m].powi(2)).sum::<f64>() - s2).abs());
        let mut e = 0.0;
        for x in 0..n {
            let lk = (0..n).map(|y| lap[x * n + y] * k[y * n + m]).sum::<f64>();
            residual = residual.max((lk - (if x == m { 1.0 } else { 0.0 }) + p).abs());
            for &y in &g.adj[x] {
                if x < y {
                    e += (k[x * n + m] - k[y * n + m]).powi(2);
                }
            }
        }
        energy = energy.max((e - a).abs());
    }
    for y in 0..n {
        let sig = (0..n)
            .flat_map(|m| g.adj[y].iter().map(move |&x| (m, x)))
            .map(|(m, x)| (k[x * n + m] - k[y * n + m]).powi(2))
            .sum::<f64>();
        sigma = sigma.max((sig - 2.0 * a).abs());
    }
    for m in [0, n / 2, n - 1] {
        let v: Vec<usize> = (0..n).filter(|&x| x != m).collect();
        let z = n - 1;
        let sub: Vec<f64> = (0..z * z).map(|i| lap[v[i / z] * n + v[i % z]]).collect();
        let inv = inverse(&sub, z);
        let mut h = vec![0.0; n];
        for i in 0..z {
            h[v[i]] = d as f64 * (0..z).map(|j| inv[i * z + j]).sum::<f64>();
        }
        for x in 0..n {
            hitting = hitting.max((h[x] - n as f64 * d as f64 * (a - k[x * n + m])).abs());
        }
        hitting = hitting.max((h.iter().sum::<f64>() / n as f64 - n as f64 * d as f64 * a).abs());
    }
    let ev = eigenvalues(&lap, n);
    assert!(ev[0].abs() < 1e-10 && ev[1] > 0.0);
    assert!(s2 <= a / ev[1] + 1e-10);
    let mut ur: f64 = 0.0;
    let mut ratio: f64 = 0.0;
    let mut identity: f64 = 0.0;
    let mut cross: f64 = 0.0;
    let mut sharp_ratio: f64 = 0.0;
    for trial in 0..6 {
        let (u, w) = if trial % 2 == 1 {
            if let Some(vertices) = triangle {
                // Unlike an arc permutation with on-site coins, this permits a
                // nonzero Re<D a,(U-D) a>, which is essential on triangles.
                (triangle_step(n, vertices, &random_unitary(3, rng)), 1)
            } else if let Some(l) = g.square {
                (plaquette_control(l, 2, rng), 2)
            } else {
                arc_control(g, rng)
            }
        } else {
            arc_control(g, rng)
        };
        ur = ur.max(unitarity(&u));
        assert!(ur < 2e-11);
        for (i, row) in u.iter().enumerate() {
            for &(j, _) in row {
                assert!(i / w == j / w || g.adj[i / w].contains(&(j / w)));
            }
        }
        let mut v: Vec<C> = (0..n * w).map(|_| rng.c()).collect();
        normalize(&mut v);
        let comm = commutator(&u, w, &k, n, &v);
        ratio = ratio.max(comm / (2.0 * a / n as f64));
        assert!(ratio < 1.0 + 1e-10);
        let ua = apply(&u, &v);
        let da = onsite(&u, w, &v);
        let re = dot(&da, &ua).re;
        if triangle_free {
            cross = cross.max((re - norm2(&da)).abs());
        }
        if g.edge_transitive {
            let scale = 2.0 * a / (d * n) as f64;
            let rhs = scale * (1.0 - re);
            identity = identity.max((comm - rhs).abs());
            let bound = scale * if triangle_free { 1.0 } else { 2.0 };
            sharp_ratio = sharp_ratio.max(comm / bound);
            assert!(sharp_ratio <= 1.0 + 1e-10);
        }
    }
    assert!(
        residual < 2e-11
            && diagonal < 2e-11
            && normspan < 2e-11
            && energy < 2e-11
            && sigma < 2e-11
            && hitting < 2e-9
            && identity < 2e-11
            && cross < 2e-11
    );
    let identity_text = if g.edge_transitive {
        format!("{identity:.3e}")
    } else {
        "NA".into()
    };
    let cross_text = if triangle_free {
        format!("{cross:.3e}")
    } else {
        "NA".into()
    };
    let sharp_text = if g.edge_transitive {
        format!("{sharp_ratio:.12e}")
    } else {
        "NA".into()
    };
    println!("GRAPH,{},{n},{d},{a:.16e},{s2:.16e},{:.16e},{residual:.3e},{diagonal:.3e},{normspan:.3e},{energy:.3e},{sigma:.3e},{hitting:.3e},{ur:.3e},{ratio:.12e},{identity_text},{cross_text},{},{triangle_free},{sharp_text}",g.name,ev[1],g.edge_transitive);
    if let Some(vertices) = triangle.filter(|_| g.edge_transitive) {
        // This state violates the triangle-free constant, while obeying the
        // general identity: Re<D a,U a> = -1/3 and Re<D a,(U-D) a> = -4/9.
        let reflection: Vec<C> = (0..9)
            .map(|i| C::r(2.0 / 3.0 - if i / 3 == i % 3 { 1.0 } else { 0.0 }))
            .collect();
        let u = triangle_step(n, vertices, &reflection);
        assert!(unitarity(&u) < 1e-12);
        let mut v = vec![C::default(); n];
        for x in vertices {
            v[x] = C::r(1.0 / 3.0_f64.sqrt());
        }
        let ua = apply(&u, &v);
        let da = onsite(&u, 1, &v);
        let re = dot(&da, &ua).re;
        let r = commutator(&u, 1, &k, n, &v) / (2.0 * a / (d * n) as f64);
        let cross = re - norm2(&da);
        assert!((re + 1.0 / 3.0).abs() < 1e-12 && (r - 4.0 / 3.0).abs() < 1e-12);
        assert!((cross + 4.0 / 9.0).abs() < 1e-12);
        println!("TRIANGLE_EXCEPTION,{},Re_Da_Ua={re:.16e},onsite_cross={cross:.16e},comm_over_triangle_free_bound={r:.16e}",g.name);
    }
    if g.edge_transitive {
        let mut shift: Sparse = vec![Vec::new(); n * d];
        for x in 0..n {
            for i in 0..d {
                let y = g.adj[x][i];
                let j = g.adj[y].iter().position(|&v| v == x).unwrap();
                shift[x * d + i].push((y * d + j, C::r(1.0)));
            }
        }
        let mut state: Vec<C> = (0..n * d).map(|_| rng.c()).collect();
        normalize(&mut state);
        let value = commutator(&shift, d, &k, n, &state);
        let ratio = value / (2.0 * a / (d * n) as f64);
        assert!((ratio - 1.0).abs() < 1e-11);
        println!(
            "PURE_SHIFT,{},comm_over_triangle_free_bound={ratio:.16e}",
            g.name
        );
    }
}
fn phi(c: usize, l: usize) -> usize {
    if c < (l + 1) / 2 {
        2 * c
    } else {
        2 * (l - c) - 1
    }
}
fn check_permutation(p: &[usize], sites: &[usize], l: usize) {
    let mut used = vec![false; p.len()];
    for (s, &t) in p.iter().enumerate() {
        assert!(t < p.len() && !used[t]);
        used[t] = true;
        let a = sites[s];
        let b = sites[t];
        assert!((a % l).abs_diff(b % l) + (a / l).abs_diff(b / l) <= 1);
    }
}
fn check_fold(l: usize, rng: &mut Rng) {
    let mut occupied = vec![false; l];
    for c in 0..l {
        let p = phi(c, l);
        assert!(p < l && !occupied[p]);
        occupied[p] = true;
    }
    let n = l * l;
    let base = 8 * n;
    let mut sites = Vec::new();
    for y in 0..l {
        for x in 0..l {
            for _ in 0..8 {
                sites.push(phi(y, l) * l + phi(x, l));
            }
        }
    }
    let mut w: Vec<usize> = (0..base).collect();
    let mut slot = vec![None; base];
    let mut slots_per_site = vec![0; n];
    let mut max_distance = 0;
    for y in 0..l {
        for x in 0..l {
            for dir in 0..4 {
                let (tx, ty, td) = match dir {
                    0 => ((x + 1) % l, y, 1),
                    1 => ((x + l - 1) % l, y, 0),
                    2 => (x, (y + 1) % l, 3),
                    _ => (x, (y + l - 1) % l, 2),
                };
                let s = (y * l + x) * 8 + 4 + dir;
                let t = (ty * l + tx) * 8 + 4 + td;
                w[s] = t;
                let a = sites[s];
                let b = sites[t];
                let distance = (a % l).abs_diff(b % l) + (a / l).abs_diff(b / l);
                max_distance = max_distance.max(distance);
                assert!(distance <= 2);
                if distance == 2 {
                    let mid = ((a / l + b / l) / 2) * l + (a % l + b % l) / 2;
                    let r = sites.len();
                    sites.push(mid);
                    slot[s] = Some(r);
                    slots_per_site[mid] += 1;
                }
            }
        }
    }
    assert!(slots_per_site.iter().all(|&x| x <= 4));
    let dim = sites.len();
    let mut q1: Vec<usize> = (0..dim).collect();
    let mut q2 = q1.clone();
    for s in 0..base {
        if let Some(r) = slot[s] {
            q1[s] = r;
            q1[r] = s;
            q2[s] = r;
            q2[r] = w[s];
        } else {
            q2[s] = w[s];
        }
    }
    check_permutation(&q1, &sites, l);
    check_permutation(&q2, &sites, l);
    for s in 0..dim {
        assert_eq!(q2[q1[s]], if s < base { w[s] } else { s });
    }
    let mut a: Vec<C> = (0..base).map(|_| rng.c()).collect();
    normalize(&mut a);
    let mut simulated = vec![C::default(); dim];
    // Include the on-site coin in the first local step, not in a third step.
    let mut after_coin = vec![C::default(); base];
    for v in 0..n {
        for j in 0..4 {
            after_coin[v * 8 + j] = a[v * 8 + j] * (-1.0);
        }
        let sum = a[v * 8 + 4..v * 8 + 8]
            .iter()
            .fold(C::default(), |s, &z| s + z);
        for j in 0..4 {
            after_coin[v * 8 + 4 + j] = sum * 0.5 - a[v * 8 + 4 + j];
        }
    }
    assert!((norm2(&after_coin) - 1.0).abs() < 2e-12);
    for s in 0..base {
        simulated[q2[q1[s]]] += after_coin[s];
    }
    let mut target = vec![C::default(); dim];
    for s in 0..base {
        target[w[s]] += after_coin[s];
    }
    let error = simulated
        .iter()
        .zip(&target)
        .map(|(&s, &t)| (s - t).abs())
        .fold(0.0_f64, f64::max);
    assert!(error < 1e-14);
    assert!(simulated[base..].iter().all(|z| z.abs2() == 0.0));
    // A marked oracle acts only on original slots of the correct folded vertex;
    // all transit states have zero amplitude at every simulated query boundary.
    for mark in [0, n / 2, n - 1] {
        for j in 0..8 {
            assert_eq!(sites[mark * 8 + j], phi(mark / l, l) * l + phi(mark % l, l));
        }
    }
    println!("FOLD,L={l},parity={},base={base},transit={},max_slots={},max_edge_length={max_distance},basis_mismatches=0,complex_coin_error={error:.3e}",if l%2==0 {"even"} else {"odd"},dim-base,slots_per_site.iter().max().unwrap());
}
fn check_log(l: usize) {
    let lf = l as f64;
    let n = lf * lf;
    let mut a = 0.0;
    let mut continuum = 0.0;
    let mut max_scaled: f64 = 0.0;
    let lo = -(l as isize / 2);
    let hi = (l as isize + 1) / 2;
    for x in lo..hi {
        for y in lo..hi {
            if x == 0 && y == 0 {
                continue;
            }
            let r2 = (x * x + y * y) as f64;
            let lam =
                4.0 * ((PI * x as f64 / lf).sin().powi(2) + (PI * y as f64 / lf).sin().powi(2));
            let f = 1.0 / (n * lam);
            let c = 1.0 / (4.0 * PI * PI * r2);
            let scaled = n * (f - c);
            assert!(scaled >= -1e-8 && scaled <= PI * PI / 48.0 + 1e-8);
            max_scaled = max_scaled.max(scaled);
            a += f;
            continuum += c;
        }
    }
    assert!(a - continuum >= -1e-9 && a - continuum <= PI * PI / 48.0 + 1e-8);
    println!("LOG,L={l},A={a:.16e},A_minus_log={:.16e},lattice_sum_remainder={:.16e},max_single_correction_times_N={max_scaled:.16e}",a-lf.ln()/(2.0*PI),a-continuum);
}
fn nontransitive_guard() {
    let mut edges = Vec::new();
    for off in [0, 4] {
        for i in 0..4 {
            for j in i + 1..4 {
                if (i, j) != (0, 1) {
                    edges.push((off + i, off + j));
                }
            }
        }
    }
    edges.extend([(0, 4), (1, 5)]);
    let g = Graph::new("regular_nontransitive".into(), 8, edges, None, false);
    let n = g.n();
    let a: Vec<f64> = g.lap().iter().map(|v| v + 1.0 / n as f64).collect();
    let inv = inverse(&a, n);
    let diagonal: Vec<f64> = (0..n).map(|i| inv[i * n + i] - 1.0 / n as f64).collect();
    let lo = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = diagonal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(hi - lo > 1e-4);
    println!("NONTRANSITIVE_GUARD,degree=3,diag_min={lo:.16e},diag_max={hi:.16e}");
}
fn main() {
    println!("GRAPH,name,N,degree,A,S2,lambda2,poisson_error,diagonal_error,S2_error,energy_error,sigma_error,hitting_error,unitarity_error,comm_fraction,sharp_identity_error,onsite_cross_error,edge_transitive,triangle_free,sharp_bound_fraction");
    let mut rng = Rng(0x169532a173f1);
    for n in [4, 5, 6, 7, 8, 9, 12] {
        check_graph(&cycle(n), &mut rng);
    }
    for n in [2, 3, 4, 5, 6, 8] {
        check_graph(&complete(n), &mut rng);
    }
    for bits in [3, 5] {
        check_graph(&cube(bits), &mut rng);
    }
    for n in [7, 8, 9, 12] {
        check_graph(&circulant(n), &mut rng);
    }
    for (l, h) in [(5, 2), (3, 5), (4, 6)] {
        check_graph(&torus(l, h), &mut rng);
    }
    for l in [3, 4, 5, 6, 7, 8] {
        check_graph(&torus(l, l), &mut rng);
    }
    nontransitive_guard();
    for l in 3..=32 {
        check_fold(l, &mut rng);
    }
    for l in [
        3, 4, 5, 8, 16, 31, 32, 63, 64, 127, 128, 256, 512, 1024, 2048,
    ] {
        check_log(l);
    }
    println!("PASS,all assertions passed");
}
