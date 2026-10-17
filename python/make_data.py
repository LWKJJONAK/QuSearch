"""Lattice constants and Tulsi's walk (Tables S2 and S3, Fig. 2c).

    python python/make_data.py [output/data]

Writes
    lattice_constants.csv  self-potential A = K(0) and norm S = ||K||_2 on d-dimensional tori
    tulsi_trajectory.csv   weighted overlap F_t / A, success probability p_t and
                           rate |F_t - F_(t-1)| / sqrt(A / 2N) after each local step
    tulsi_summary.csv      one row per L: peak step, overlap, rates and both lower bounds

Tulsi's walk: four coin directions and one control qubit per vertex, cos(delta) = 1/sqrt(1 + 8A),
uniform initial state, marked vertex at the origin. One iteration is the marked reflection
followed by one local step. All amplitudes are real, so F is real.
"""
import csv
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/data"
os.makedirs(OUT, exist_ok=True)


def green(L, d):
    """Zero-mean Green function of the d-dimensional L-torus Laplacian (2d I - adjacency)."""
    k = np.arange(L)
    s = 4 * np.sin(np.pi * k / L) ** 2
    lam = sum(np.meshgrid(*([s] * d), indexing="ij"))
    inv = np.zeros_like(lam)
    inv[lam > 1e-300] = 1 / lam[lam > 1e-300]
    return np.real(np.fft.ifftn(inv))


# ------------------------------------------------------------------ lattice constants
rows = []
for d, Ls in ((1, [16, 64, 256, 1024, 4096]),
              (2, [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
              (3, [8, 16, 32, 64, 128])):
    for L in Ls:
        K = green(L, d)
        rows.append(dict(dimension=d, L=L, N=L ** d, A=K.flat[0], S=np.sqrt((K ** 2).sum())))
with open(f"{OUT}/lattice_constants.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})


# ------------------------------------------------------------------ Tulsi's walk
def tulsi(L):
    K = green(L, 2)
    N = L * L
    A = K[0, 0]
    S = np.sqrt((K ** 2).sum())
    c = 1 / np.sqrt(1 + 8 * A)
    s = np.sqrt(1 - c * c)
    parked = np.zeros((L, L, 4))
    walk = np.full((L, L, 4), 1 / (2 * L))
    a_walk = walk.copy()                        # unmarked run is stationary
    B0 = (A - K)[..., None]                     # B_0(x) = A - K(x)

    def F(walk):                                # translation invariance: F = <a, B_0 b^0>
        return float(np.sum(a_walk * B0 * walk))

    lim = np.sqrt(A / (2 * N))
    tmax = int(np.ceil(1.5 * 0.5 * np.pi * L * np.sqrt(2 * A + s * s / (4 * c * c))))
    Fprev = F(walk)
    out = []
    for t in range(1, tmax + 1):
        proj = 0.5 * (-s * parked[0, 0].sum() + c * walk[0, 0].sum())
        parked[0, 0] += s * proj                # marked reflection I - 2|delta><delta| at the origin
        walk[0, 0] -= c * proj
        assert abs(F(walk) - Fprev) < 1e-12    # queries leave F unchanged
        parked = -parked                        # local step: diag(-I, flip-flop shift x Grover coin)
        g = 0.5 * walk.sum(-1, keepdims=True) - walk
        new = np.empty_like(walk)
        new[..., 1] = np.roll(g[..., 0], 1, 0)
        new[..., 0] = np.roll(g[..., 1], -1, 0)
        new[..., 3] = np.roll(g[..., 2], 1, 1)
        new[..., 2] = np.roll(g[..., 3], -1, 1)
        walk = new
        assert abs(float((parked ** 2).sum() + (walk ** 2).sum()) - 1) < 1e-9
        Fn = F(walk)
        p = float((parked[0, 0] ** 2).sum() + (walk[0, 0] ** 2).sum())
        out.append((t, Fn / A, p, abs(Fn - Fprev) / lim))
        Fprev = Fn
    return A, S, out


summ = []
with open(f"{OUT}/tulsi_trajectory.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["L", "t", "F_over_A", "p", "rate"])
    for L in [16, 32, 64, 128, 256, 512, 1024]:
        A, S, out = tulsi(L)
        N = L * L
        arr = np.array(out)
        i = int(np.argmax(arr[:, 2]))
        T, Fa, p = int(arr[i, 0]), arr[i, 1], arr[i, 2]
        # F decreases monotonically up to the peak, so mean_rate below is the mean of |dF|
        assert np.diff(np.concatenate(([1.0], arr[:i + 1, 1]))).max() <= 1e-12
        num = (1 - np.sqrt(1 - p) - 1 / L) * L * A - S
        t_theorem, t_min = num / np.sqrt(2 * A), num / np.sqrt(A / 2)
        mean_rate = (1 - Fa) * A / (T * np.sqrt(A / (2 * N)))
        need = A * (1 - np.sqrt(1 - p) - 1 / L) - S / L     # drop the theorem requires
        summ.append(dict(L=L, N=N, A=A, S=S, T=T, p=p, F_T_over_A=Fa,
                         max_rate=arr[:i + 1, 3].max(), mean_rate=mean_rate,
                         drop=(1 - Fa) * A / need,
                         T_theorem=t_theorem, T_min=t_min, T_over_T_min=T / t_min,
                         T_over_LsqrtlnL=T / (L * np.sqrt(np.log(L)))))
        for r in out:
            w.writerow([L, r[0], f"{r[1]:.10g}", f"{r[2]:.10g}", f"{r[3]:.10g}"])
        print(f"L={L}: T={T}, p={p:.4f}, T_min={t_min:.2f}, T/T_min={T / t_min:.2f}", flush=True)
with open(f"{OUT}/tulsi_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0]), lineterminator="\n")
    w.writeheader()
    for r in summ:
        w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
