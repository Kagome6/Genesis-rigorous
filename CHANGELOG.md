# Changelog

All notable scientific and engineering milestones of **Project Genesis-Rigorous** are documented in this file.

## [1.0.1] - 2026-11-25

### Fixed (External Peer-Review & Mathematical Audit Overhaul)
- **Exact $(D-1)/D$ Isotropic 1-Loop Factor in $O(N^3)$ Engine (`genesis_rigorous/matrix_native.py`)**:
  - Corrected the diagonal Yang–Mills Hessian isotropic average factor from $g^2 \Delta_Y$ to $\frac{D-1}{D} g^2 \Delta_Y$ (reflecting $[Y_I, Y_I] = 0$, i.e. $2/3$ in $D=3$ and $5/6$ in $D=6$) across `om_su`, `E_fluc_pot`, and the envelope-theorem gradient `dE_bos`.
  - Closed the `T13` audit blind spot in `genesis_rigorous/verification.py` by adding an explicit non-zero $Y$ Hessian trace identity check $\mathrm{Tr}(H^{(2)}_{\mathrm{YM}}) = (D-1)g^2 \mathrm{Tr}(\Delta_Y)$ verified to `1.87e-16` alongside the exact $Y=0$ HFB match.
- **Far-Field Causal Light-Cone Fit (`genesis_rigorous/experiments/phase3a_dynamics.py`, `genesis_cli.py`)**:
  - Restricted the Phase 3A causal wavefront velocity fit to the far-field window $\theta \in [85^\circ, 160^\circ]$ outside the finite-$N$ noncommutative wave-packet overlap plateau ($\theta \le 80^\circ$), revealing a sharper linear light cone ($r = 0.9983$, $c_{\mathrm{onset}} = 1.483\;\mathrm{rad}/t$) honestly below the bare Yang–Mills dispersion bound $g\alpha = 2.808\;\mathrm{rad}/t$.
  - Synchronized default real-time step count to `n_steps = 450` across CLI and experiment module.
- **Coulomb-Branch Matrix Higgs Terminology & Mode-Tracked Spectrum (`genesis_rigorous/experiments/phase3c_gauge.py`)**:
  - Replaced the imprecise "Spontaneous Symmetry Breaking" plot label with "Coulomb-Branch Matrix Higgs Splitting (External Brane Separation $v$)" since $+\frac{1}{2}\mu^2 \mathrm{Tr}(X_4^2)$ ($\mu=1$) is confining rather than flat/Mexican-hat.
  - Eliminated branch-crossing zigzag artifacts in the Phase 3C mode-spectrum plot by explicitly tracking diagonal $U(1)^2$ and off-diagonal $W^\pm$ lowest and first KK branches.
- **Figure–Text–Table Consistency Across All 7 Publication Figures**:
  - Fixed Phase 1 Panel 2 caption (`Bright Yellow Ring` for `inferno_r`) and added raw 4-block asymmetric $S_{\mathrm{EE}}(M)$, symmetrized $\bar{S}_{\mathrm{EE}}(M)$, and raw $S_{\mathrm{TN}}(M)$ curves to Panel 5.
  - Fixed Stage 3 Panel 6 to plot the exact $M=4$ blind hemisphere $\cos\theta$ (`-0.585 → -0.903`) matching Table 5, and added physical discussion of why local metric density $\sqrt{\det g_N}$ drives $+27.8\%$ North-cap $S_{\mathrm{EE}}$ growth despite embedded perimeter contraction.
  - Re-ran Phase 3B ($N=16, 20, 24, 32$) and Phase 3D ($D=6, N=16$ & chiral $q=0\dots 3$) with the corrected $(D-1)/D$ engine and updated all dashboard titles and LaTeX captions to match the exact rendered panels.
- **Packaging & Documentation Fixes**:
  - Fixed `pyproject.toml` (`readme = "README.md"`, `py-modules = ["genesis_cli"]`, `version = "1.0.1"`) so `pip install .` and `pip install -e .` succeed cleanly from the GitHub layout.
  - Fixed relative link in `docs/ARCHITECTURE.md` and synchronized CLI commands in `paper/genesis_rigorous_report.tex` Section 8.

## [1.0.0] - 2026-11-25

### Added
- **Unified Package & CLI (`genesis_rigorous/`, `genesis_cli.py`, `pyproject.toml`)**:
  - Consolidated all verified physics cores and experiment runners under a single modular Python package with subcommands `verify`, `phase1`, `gravity`, `realtime`, `large-n`, `gauge`, `chiral`, and `suite`.
  - Added 17-test master mathematical and gauge-invariance audit suite (`genesis_rigorous/verification.py` and `tests/test_genesis_audit.py`).
- **Phase 1 (Emergent Geometry & Blind Minimal-Area Law, $N=8, 10, 12$)**:
  - Exact HFB 4th-order Wick expansion + exact 2-flavor BdG link-fermion backreaction + unclipped Barzilai–Borwein/Armijo isotropic multi-scale annealing.
  - 100% $U(N)$ gauge-invariant Point-Probe Dirac geometry and exact perturbation-theory Provost–Vallée Riemannian quantum metric.
  - Blind $\mathfrak{su}(N)$ minimal-entanglement search recovering exact 2D spherical caps and area-law scaling ($r(S_{\text{TN}}) > 0.996$).
- **Phase 2 / Stage 3 (Emergent Gravity from Adjoint Dirac Matter, $N=8, 10$)**:
  - Exact $4N^2\times 4N^2$ Hermitian adjoint Dirac operator $\mathcal{D}_{\text{adj}}(Y)$ with singularity-free Daleckii–Krein rationalized divided-difference gradient.
  - Demonstrated local positive Gaussian curvature concentration ($r > 0.999$), Provost–Vallée area-element response, North-vs-South entanglement first-law shift, and Hubeny-like blind minimal-surface gravitational repulsion.
- **Phase 3A (Two-Body Emergent Attraction & Real-Time Causal Light Cone)**:
  - Verified curvature-mediated two-body Yukawa attraction ($V_{\text{grav}}(25^\circ) = -8.77$ vs frozen-geometry control `1.7e-5`) and symplectic Velocity-Verlet real-time curvature wave propagation ($c_{\text{ang}} = 2.534\;\text{rad}/t$ vs theoretical $g\alpha = 2.808\;\text{rad}/t$).
- **Phase 3B ($O(N^3)$ Matrix-Native Engine & Large-$N$ Continuum Limit, $N=16\dots 32$)**:
  - Eliminated $O(N^6)$ `f_abc` tensor via direct matrix commutators and Kronecker-product Laplacian (537× memory reduction; $N=32$ in `8.2 MB` VRAM).
- **Phase 3C (Emergent $U(2)$ Gauge Bosons & Matrix Higgs Mechanism)**:
  - Demonstrated exact 4-fold $U(2)$ degeneracy at $v=0$, exact $U(2)\to U(1)\times U(1)$ Higgs mass splitting $\Delta\omega_W^2 = g^2 v^2$ (`MaxErr = 2.21e-13`), 2 eaten Goldstone modes (`436 → 434`), and non-Abelian $SU(2)$ quartic self-interaction.
- **Phase 3D (4D Emergent Space $S^2\times S^2$ & Atiyah–Singer Chiral Fermion Generations)**:
  - Spontaneous emergence of a Rank-4 quantum metric manifold from $D=6, N=16$ random GUE matrices (`[0, 0, 0.00554, 0.00554, 0.00761, 0.00761]`).
  - Exact Ginsparg–Wilson / GKP chiral Dirac operator verifying $q = 0, 1, 2, 3$ topologically protected massless chiral fermion generations (`|E_0| <= 7.0e-16` even under strong random matrix noise).
- **Documentation & Publication Package**:
  - Added `README.md`, `CITATION.cff`, `LICENSE`, `.gitignore`, `requirements.txt`, `docs/ARCHITECTURE.md`, and Overleaf-ready LaTeX technical report (`paper/genesis_rigorous_report.tex`, `paper/references.bib`).

### Fixed (Silent-Bug Mathematical Audits)
- Fixed `einsum` subscript collision (`"cdab,sdbc->csac"`) in Term 2 of `gauss_casimir_and_boson_entropy`.
- Added `.contiguous()` safeguards prior to `torch.kron` for rotated Hermitian matrices.
- Resolved HFB Hohenberg–Martin Goldstone-mode pollution of the subregion Gauss-law Casimir via exact $U(N)$-covariant Coulomb-transverse projection $P_{\text{phys}}$ (`Cas_Term1(M=N) = 1.47e-27`).
- Replaced `eigh(Sx_in)` with Cholesky similarity $L_x^T S_{p,\text{in}} L_x$ to eliminate cuSOLVER degeneracy failures on isotropic vacuum states.
- Separated the decoupled $U(1)$ trace mode ($k=0$) from interacting $SU(N)$ modes in the $O(N^3)$ 1-loop gap equation (`dE_vac = 1.42e-14` match vs exact HFB).
