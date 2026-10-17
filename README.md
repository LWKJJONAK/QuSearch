# QuSearch

Code and data for the paper **Electrostatic Limits on Quantum Spatial Search** by Weikang Li. The
numerical results illustrate the theorems; no proof depends on them.

## Usage

You need Python 3 with NumPy and a [Rust toolchain](https://rustup.rs) (1.96 or later), which
links with the system C linker `cc`; the Rust programs use the standard library only.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh data      # regenerate data/ into output/data and compare with data/
./reproduce.sh checks    # finite-size checks of the Supplemental Material
```

`data/` is never modified. `./reproduce.sh data` takes a few minutes and 1–2 GB of memory, mostly
for Tulsi's walk on the 1024 × 1024 torus. `python/compare.py` accepts floating-point differences
up to 1e-9. The supplied data were generated with Python 3.12, NumPy 2.5 and Rust 1.98.
Regenerating them on Linux x86_64 and macOS arm64 with NumPy 1.26–2.5 and Rust 1.96–1.98 passes
the comparison; most files come out byte for byte identical, and the others differ only by
rounding errors far below this tolerance.

## Contents

| Paper | Data | Program |
|---|---|---|
| Table S2 | `lattice_constants.csv` | `python/make_data.py` |
| Fig. 1c; A − (ln L)/2π ≈ 0.0488 | `spectral.csv` | `src/torus.rs` |
| Table S3, Fig. 2c and Tulsi's walk in the main text | `tulsi_summary.csv`, `tulsi_trajectory.csv` | `python/make_data.py` |
| Second simulation of Tulsi's walk (Sec. S9) | `tulsi_independent.csv` | `src/torus.rs` |
| Table S4, Fig. 2b and the marked-set results of the main text and Sec. S9 | `marked_equilibrium.csv`, `marked_summary.csv`, `marked_patterns.csv` | `src/marked_equilibrium.rs` |
| Consistency checks (Sec. S9) | `output/checks/` | `src/check_bound.rs`, `src/check_graphs.rs` |

The columns of every file are described in [`data/README.md`](data/README.md). Figure-drawing
code is not included. The data behind Figs. 1c and 2b–c are in `data/`; the potential maps of
Figs. 1a and 2a follow directly from the definitions, and Fig. 2a uses the `periodic` and
`cluster` sets with L = 64 and k = 16.

## Checks

Every program aborts if one of its assertions fails, so a complete run confirms the statements
of Sec. S9.

- `make_data.py`: queries change F by less than 1e-12 and the norm stays within 1e-9 of one at
  every step of Tulsi's walk, and F decreases monotonically up to the peak.
- `marked_equilibrium.rs`: charge normalization and positivity, the constant value on the marked
  set, zero mean, the Poisson equation, the energy identity and the maximum principle for all 261
  sets; the quotient-torus formulas for periodic subgrids; and a direct solution of the Dirichlet
  problem on the 8 × 8 torus, which reproduces A_M and the hitting times.
- `check_bound.rs`: on L × L tori with L = 5, 8, 12 and 16, time- and position-dependent local
  steps and queries preserve the norm, leave F unchanged at queries and obey the speed limit at
  every local step.
- `check_graphs.rs`: the identities and inequalities of Secs. S2, S3, S5 and S7. On 28 pairwise
  non-isomorphic vertex-transitive graphs (`torus_5x2` is the prism) it checks the potential
  identities, S² ≤ A/λ₂ and the hitting times; on the 21 edge-transitive ones, the commutator
  identity; and the triangle-free bound, which holds on 15 of these and fails on the other six.
  It also checks the folded routing for L = 3 to 32 and the bounds on A of Sec. S4. `NA` marks a
  check whose symmetry or triangle-free hypothesis does not hold. The printed residuals are
  below 1e-12.

## Citation

If you use this code or data, please cite the paper:

> Weikang Li, *Electrostatic Limits on Quantum Spatial Search* (2026).

## License

[MIT](LICENSE)
