# Changelog

All notable scientific, mathematical, and engineering milestones of **Project Genesis-Rigorous** are documented in this file.

## [1.0.2] - 2026-11-27

### Added (Resolving All Three Core Gaps Identified in Independent Audit of v1.0.1)
- **Strictly Traceless $\mathfrak{su}(N)$ Unit-Radius Link Entanglement (`genesis_rigorous/exact_hfb.py`, `experiments/v102_entanglement_scaling.py`, CLI `entanglement-scaling`)**:
  - Removed the geometry-blind $k=0$ identity mode $E_0 = I_N/\sqrt{N}$ prior to BdG diagonalization via an explicit $N^2-1$ orthonormal traceless basis and $U(N)$-invariant unit-radius normalization $\tilde{Y}_I = Y_I / R_{\mathrm{rms}}$.
  - Added full $N$-scaling benchmarks at fixed $M = N/2$ across $N = 6, 8, 10, 12, 16, 20, 24$ with AICc/BIC model selection against same-norm random GUE and aligned commuting-matrix controls: fuzzy $S^2$ obeys $O(N)$ matrix-boundary scaling ($N^{0.834 \pm 0.005}$, $\Delta\mathrm{BIC}=+24.51$ over $N^2$) whereas same-norm GUE obeys $O(N^2)$ matrix-volume scaling ($N^{1.933 \pm 0.024}$, $5.00\times$ larger at $N=24$). Mirrored in bosonic HFB entropy ($N^{1.37}$ vs.\ $N^{2.31}$, $6.32\times$ separation at $N=12$).
- **Exact $U(N)$-Singlet Gauged Lattice HMC Engine (`genesis_rigorous/hmc_singlet.py`, `experiments/v102_phase_transition_hmc.py`, CLI `phase-hmc`)**:
  - Implemented a strictly $U(N)$-singlet finite-temperature lattice Hybrid Monte Carlo sampler with dynamical unitary Polyakov-loop holonomy $U \in U(N)$, loop-free 3-commutator analytical forces, UV-finite virial energy estimator, and exact discrete Schwinger–Dyson stationarity diagnostics ($3 N^2 L$).
  - Mapped the true first-order transition above the classical $\nu_c = \frac{3}{2}\mu$ (distinct from the spinodal $\nu_{\mathrm{spin}} = \sqrt{2}\mu$), resolving $\nu_c^{\mathrm{quantum}} \approx 1.65\text{–}1.68$ at $g=0.30, N=4$ with $0.6\%\text{–}0.8\%$ HFB–HMC radius match on the sphere branch and an exact decomposition of the $+18.45$ energy offset ($+4.64$ Hohenberg–Martin gauge-mode ZPE $+ 15.74$ lattice $a=0.25$ dispersion).
  - Swept into strong coupling ($g \in [0.25, 1.45]$ at $\nu=2.20, N=4$), directly capturing non-perturbative fuzzy-sphere melting ($\mathrm{AlgErr}: 0.157 \to 10.53$), Polyakov-loop confinement ($|\mathrm{Tr}\,U|/N: 0.983 \to 0.220$), and a $48.2\%$ mean-field HFB radius breakdown at $g=0.85$.
- **Matched-Energy Generic Elastic Membrane Control (`experiments/v102_membrane_control.py`, CLI `membrane-control`)**:
  - Directly contrasted localized adjoint-Dirac packets against a generic local radial potential $\mathrm{Tr}(\hat{\rho} Y^2)$ matched to identical source energy ($E_{\mathrm{src}} = 1254.38$ at $N=8$).
  - Proved that short-range two-body attraction ($V(25^\circ)=-14.18$ for $\mathrm{Tr}(\hat{\rho}Y^2)$ vs.\ $-8.77$ for Dirac) is universal to elastic-shell overlap, while single-body curvature sign distinguishes prolate commutator-tension pinching ($+11.3\%$) from oblate radial indentation ($-9.8\%$).
- **Expanded 20-Test Automated Audit Suite (`T01`–`T20`) (`genesis_rigorous/verification.py`, `tests/test_genesis_audit.py`)**:
  - Added `T18` (traceless $\mathfrak{su}(N)$ BdG projector trace-zero & $U(N)$ covariance), `T19` (analytical HMC forces $F_X$ and unitary $F_U$ vs.\ central finite differences), and `T20` (exact free-oscillator thermal expectation & discrete Schwinger–Dyson identity check). All 20 tests pass in $\sim 3.6\text{ s}$.
- **Zero-Cheat Visual Observatories (`experiments/v102_spacetime_viewer.py`, `experiments/v102_live_observatory.py`, CLI `watch` & `watch-quantum`)**:
  - Added classic 5-panel observatory (`--mode birth`, `melting`, `emergence`) with zoomed, facet-shaded static-camera 3D Point-Probe Dirac shells, glowing `inferno` $z=0$ cross-sections, $R^2$ + Polyakov-loop curves, and normalized QFIM ($4 g_{\mathrm{PV}}$) rank-2 tracking (removing the legacy `Phi_TN` panel).
  - Added real-time interactive $\nu / g$ slider HMC observatory (`watch-quantum`).
- **Peak VRAM Instrumentation & Honest Complexity Documentation (`genesis_rigorous/safety.py`, `matrix_native.py`)**:
  - Added `reset_peak_vram()` / `get_peak_vram_mb()` and corrected all complexity documentation for `MatrixNativeQuantumCore` ($O(N^3)$ classical forces, $O(N^4)$ Laplacian storage, $O(N^6)$ dense Laplacian eigensolve).
- **Updated Publication Manuscript & Zenodo Archive (`paper/genesis_rigorous_report_v1.0.2.tex`, `paper/references.bib`, `README.md`, `CITATION.cff`, `docs/ARCHITECTURE.md`)**.

## [1.0.1.2] - 2026-11-26

### Changed (Honest Scientific Reframing of v1.0.1 Manuscript)
- Produced a standalone corrected report for the archived `v1.0.1` codebase addressing every mathematical, statistical, complexity, and citation point raised in external adversarial audit without modifying `v1.0.1` code (retracting the $k=0$ area-law claim, distinguishing $\nu_{\mathrm{spin}}=\sqrt{2}\mu$ from $\nu_c=1.5\mu$, reframing Stage 3/3A as noncommutative elastic membrane response, and verifying 28/28 literature references).

## [1.0.1] - 2026-11-25

### Fixed (External Peer-Review & Mathematical Audit Overhaul)
- Corrected isotropic 1-loop Yang–Mills factor to $\frac{D-1}{D} g^2 \Delta_Y$ in `matrix_native.py` and verified `Tr(H2_YM) = (D-1) g^2 Tr(Delta_Y)` in `T13`.
- Restricted Phase 3A wavefront velocity fit to far-field window $\theta \in [85^\circ, 160^\circ]$ ($c_{\mathrm{onset}} = 1.483\;\mathrm{rad}/t$, $r=0.9983$).
- Reframed Phase 3C as Coulomb-branch Matrix Higgs splitting with branch-tracked mode spectra.
- Synchronized Phase 1, Stage 3, Phase 3B, and Phase 3D figures, captions, and CLI defaults.

## [1.0.0] - 2026-11-25

### Added
- Initial unified PyTorch/CUDA package (`genesis_rigorous/`), CLI (`genesis_cli.py`), 17-test verification suite, and Phase 1–3D experiment modules.
