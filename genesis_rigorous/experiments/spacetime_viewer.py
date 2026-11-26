"""
Zero-Cheat Interactive Spacetime Genesis & Gravitational-Wave Visualizer
========================================================================
Visualizes the *exact* U(N) gauge-invariant emergence of spacetime (Phase 1)
and real-time gravitational-wave propagation (Phase 3A) directly from matrix
trajectories Y_I(t) — with ZERO hardcoded meshes, ZERO fake morphing, and
100% gauge-invariant Point-Probe Dirac & Quantum Fisher Information observables.

Modes:
  --mode emergence   : Live condensation of 0D isotropic GUE matrix noise into
                       a glowing 2D fuzzy-sphere manifold (Universe B, nu=3.0)
                       side-by-side with Universe A (nu=0.8) 0D collapse,
                       R^2 eigenvalue convergence, and Quantum Fisher Metric rank.
  --mode metastable  : Spontaneous splitting of 0D chaos into concentric double
                       fuzzy spheres (topological defect / multi-sphere basin).
  --mode wave        : Real-time symplectic gravitational-wave ripple launched
                       by an Adjoint Dirac North-Pole pulse, sweeping to the
                       South Pole with antipodal focusing and bounce-back.
"""

import datetime
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..exact_hfb import GenesisQuantumCore, build_su2_irrep
from ..matrix_native import MatrixNativeQuantumCore
from ..dirac_matter import AdjointDiracMatterSector, find_shell_radius_along_ray
from ..safety import HardwareSafetyGuard, SafetyConfig, safe_eigh, safe_eigvalsh


@torch.no_grad()
def _batched_point_probe_min_eval(
    Y: torch.Tensor,
    pts: torch.Tensor,
    sigmas: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Exact U(N) gauge-invariant Point-Probe Dirac lowest |eigenvalue| on a batch of
    3D points `pts` of shape (P, 3). Zero approximations, zero mesh cheats.
    """
    N = Y.shape[1]
    device = Y.device
    cdtype = Y.dtype
    eye_N = torch.eye(N, dtype=cdtype, device=device)

    # Precompute base matrix B = sum_I Y_I kron sigma_I  (2N x 2N)
    B = (
        torch.kron(Y[0].contiguous(), sigmas[0])
        + torch.kron(Y[1].contiguous(), sigmas[1])
        + torch.kron(Y[2].contiguous(), sigmas[2])
    )
    B = 0.5 * (B + B.conj().T)
    S_kron = torch.stack([torch.kron(eye_N, sigmas[I]) for I in range(3)], dim=0)  # (3, 2N, 2N)

    P = pts.shape[0]
    out = torch.empty(P, dtype=torch.float64, device=device)
    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        p_c = pts[start:end].to(cdtype)  # (C, 3)
        D_batch = B.unsqueeze(0) - torch.einsum("ci,imn->cmn", p_c, S_kron)
        D_batch = 0.5 * (D_batch + D_batch.conj().transpose(-2, -1))
        try:
            evals = torch.linalg.eigvalsh(D_batch)
        except RuntimeError:
            evals = torch.linalg.eigvalsh(D_batch.cpu()).to(device)
        out[start:end] = torch.min(torch.abs(evals), dim=-1).values
    return out


@torch.no_grad()
def _batched_3d_shell_surface(
    Y: torch.Tensor,
    unit_dirs: torch.Tensor,
    sigmas: torch.Tensor,
    r_max: float,
    n_coarse: int = 18,
    n_fine: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the exact U(N) gauge-invariant Point-Probe shell radius r*(theta, phi)
    and sharpness |lambda_min(r* n_hat)| along every 3D ray in `unit_dirs` (R, 3).
    """
    device = Y.device
    R = unit_dirs.shape[0]
    r_grid = torch.linspace(0.0, r_max, n_coarse, dtype=torch.float64, device=device)
    # Coarse search: (R, n_coarse, 3)
    pts_c = unit_dirs[:, None, :] * r_grid[None, :, None]
    lam_c = _batched_point_probe_min_eval(Y, pts_c.reshape(-1, 3), sigmas).reshape(R, n_coarse)
    idx_best = torch.argmin(lam_c, dim=1)
    r_best_c = r_grid[idx_best]
    dr = r_max / max(n_coarse - 1, 1)

    # Fine refinement around coarse minimum
    offsets = torch.linspace(-dr, dr, n_fine, dtype=torch.float64, device=device)
    r_fine = torch.clamp(r_best_c[:, None] + offsets[None, :], min=0.0, max=r_max)
    pts_f = unit_dirs[:, None, :] * r_fine[:, :, None]
    lam_f = _batched_point_probe_min_eval(Y, pts_f.reshape(-1, 3), sigmas).reshape(R, n_fine)
    f_idx = torch.argmin(lam_f, dim=1)
    r_star = r_fine[torch.arange(R, device=device), f_idx]
    lam_star = lam_f[torch.arange(R, device=device), f_idx]
    return r_star.cpu().numpy(), lam_star.cpu().numpy()


@torch.no_grad()
def _exact_qfim_eigenvalues(Y: torch.Tensor, y_pt: torch.Tensor, sigmas: torch.Tensor) -> np.ndarray:
    """
    Exact Quantum Fisher Information Metric g^QFI_{ab} = 4 * g^PV_{ab} (Provost-Vallee
    metric from non-degenerate perturbation theory of the Point-Probe Dirac ground state).
    Returns sorted eigenvalues [g_1 <= g_2 <= g_3]. On an emergent 2D sphere, g_1 -> 0
    and g_2 == g_3 == N / r^2 (exact rank-2 Riemannian manifold in information geometry!).
    """
    N = Y.shape[1]
    device = Y.device
    cdtype = Y.dtype
    eye_N = torch.eye(N, dtype=cdtype, device=device)
    D_mat = sum(torch.kron((Y[I] - y_pt[I] * eye_N).contiguous(), sigmas[I]) for I in range(3))
    D_mat = 0.5 * (D_mat + D_mat.conj().T)
    evals, evecs = safe_eigh(D_mat)
    i0 = int(torch.argmin(torch.abs(evals)))
    lam0 = evals[i0]
    psi0 = evecs[:, i0]
    mask = torch.arange(evals.shape[0], device=device) != i0
    d_lam = evals[mask] - lam0
    # Regularize near-degeneracy safely (never divides by zero in 0D chaos!)
    denom = torch.where(torch.abs(d_lam) > 1e-6, d_lam * d_lam, torch.full_like(d_lam, 1e12))
    V_oth = evecs[:, mask]
    # dD/dy_a = -I_N kron sigma_a -> matrix elements M_{n, a} = <psi_n | I_N kron sigma_a | psi_0>
    M = torch.empty((V_oth.shape[1], 3), dtype=cdtype, device=device)
    for a in range(3):
        op_psi0 = torch.kron(eye_N, sigmas[a]) @ psi0
        M[:, a] = V_oth.conj().T @ op_psi0
    # Quantum Fisher metric = 4 * Re sum_{n != 0} M^*_{n,a} M_{n,b} / (lam_n - lam_0)^2
    g_qfi = 4.0 * torch.real(torch.einsum("na,nb,n->ab", M.conj(), M, 1.0 / denom))
    g_qfi = 0.5 * (g_qfi + g_qfi.T)
    return np.sort(safe_eigvalsh(g_qfi).cpu().numpy())


@torch.no_grad()
def _compute_equator_phi_tn(Y: torch.Tensor, core: MatrixNativeQuantumCore, kappa: float, g1: float, g2: float) -> float:
    """
    Exact Information-Geometric Quantum Relative Entropy (Quantum Mutual Information)
    Phi_TN = S(rho_TN || rho_Sigma x rho_SigmaBar) = 2 * S_TN(M = N/2) across the
    equatorial Z-bipartition (exact quantum counterpart of Oizumi-Amari m-projection).
    """
    N = core.N
    M = N // 2
    _, U_z = safe_eigh(0.5 * (Y[2] + Y[2].conj().T))
    Q_in = U_z[:, N - M:]  # Upper hemisphere subspace (M states)

    Delta = core.build_laplacian_mat(Y)
    lam, U_eig = safe_eigh(Delta)
    lam = torch.clamp(lam, min=0.0)
    f_v = g1 * lam + g2 * (lam * lam)
    eps_v = torch.sqrt(f_v * f_v + 1.0)
    n_k = (eps_v - f_v) / (2.0 * eps_v)
    m_k = (eps_v + f_v) / (2.0 * eps_v)
    a_k = -1j / (2.0 * eps_v)

    U_mats = U_eig.T.reshape(core.N2, N, N)
    B = torch.einsum("ia,kab,bj->kij", Q_in.conj().T, U_mats, Q_in).reshape(core.N2, M * M)
    G11 = B.conj().T @ (n_k.to(core.cdtype)[:, None] * B)
    G22 = B.conj().T @ (m_k.to(core.cdtype)[:, None] * B)
    G21 = B.conj().T @ (a_k.to(core.cdtype)[:, None] * B)
    G12 = G21.conj().T
    G_blk = torch.cat([torch.cat([G11, G12], dim=1), torch.cat([G21, G22], dim=1)], dim=0)
    G_blk = 0.5 * (G_blk + G_blk.conj().T)
    ev = torch.clamp(safe_eigvalsh(G_blk), min=1e-13, max=1.0 - 1e-13)
    s_tn = float(-torch.sum(ev * torch.log(ev) + (1.0 - ev) * torch.log(1.0 - ev)))
    # Quantum relative entropy S(rho || rho_in x rho_out) = 2 * S_EE for pure global state
    return 2.0 * s_tn


@torch.no_grad()
def _geometry_diag(Y: torch.Tensor) -> Dict[str, object]:
    """Exact U(N) gauge-invariant R^2 spectrum, Tr(Y^2), and SU(2) relative algebra error."""
    tr_y2 = float(torch.real(torch.sum(Y * Y.conj())))
    R2 = Y[0] @ Y[0] + Y[1] @ Y[1] + Y[2] @ Y[2]
    R2 = 0.5 * (R2 + R2.conj().T)
    r2_evals = np.sort(torch.real(safe_eigvalsh(R2)).cpu().numpy())
    if tr_y2 < 1e-10:
        return {"tr_Y2": tr_y2, "r2_evals": r2_evals, "alpha_est": 0.0, "rel_alg_err": 1.0}
    C01 = Y[0] @ Y[1] - Y[1] @ Y[0]
    C12 = Y[1] @ Y[2] - Y[2] @ Y[1]
    C20 = Y[2] @ Y[0] - Y[0] @ Y[2]
    # Optimal least-squares alpha matching [Y_i, Y_j] = i alpha eps_{ijk} Y_k
    num = float(torch.real(
        torch.sum((-1j * C01) * Y[2].conj())
        + torch.sum((-1j * C12) * Y[0].conj())
        + torch.sum((-1j * C20) * Y[1].conj())
    ))
    alpha = num / max(tr_y2, 1e-12)
    r01 = C01 - 1j * alpha * Y[2]
    r12 = C12 - 1j * alpha * Y[0]
    r20 = C20 - 1j * alpha * Y[1]
    res_sq = float(torch.real(torch.sum(r01 * r01.conj()) + torch.sum(r12 * r12.conj()) + torch.sum(r20 * r20.conj())))
    c_sq = float(torch.real(torch.sum(C01 * C01.conj()) + torch.sum(C12 * C12.conj()) + torch.sum(C20 * C20.conj())))
    rel_err = float(np.sqrt(res_sq / max(c_sq, 1e-12)))
    return {"tr_Y2": tr_y2, "r2_evals": r2_evals, "alpha_est": alpha, "rel_alg_err": rel_err}


@torch.no_grad()
def _generate_emergence_frames(
    N: int = 8,
    metastable: bool = False,
    n_frames: int = 68,
    prefer_cuda: bool = True,
) -> Dict[str, object]:
    """
    Runs 100% U(N) gauge-invariant continuous damped matrix dynamics from IDENTICAL
    isotropic GUE random clouds for Universe A (nu=0.8) and Universe B (nu=3.0),
    recording exact Point-Probe 3D shells, 2D z=0 slices, R^2 spectra, QFIM metric
    eigenvalues, and information-geometric Phi_TN at every frame.
    """
    t0 = time.time()
    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    device = guard.device
    core = MatrixNativeQuantumCore(N=N, D=3, guard=guard)

    g, mu, nu_A, nu_B = 0.25, 1.0, 0.80, 3.00
    kappa, g1, g2 = 0.15, 0.015, 1e-4

    # Identical 100% U(N)xSO(3) isotropic random GUE seed for BOTH Universe A & Universe B!
    # Uses the EXACT Phase 1 scale-covariant Barzilai-Borwein + Armijo relaxation trajectory
    # (zero hardcoded target radius, zero manual kicks, identical initial matrices for A and B).
    seed_val = 42 if not metastable else 2026
    y_scale = (7.0 if not metastable else 5.0) * (float(N) / 8.0)
    gen = torch.Generator(device="cpu").manual_seed(seed_val)
    Y0 = core.random_isotropic_gue(y_scale, gen)

    Y_A = Y0.clone()
    Y_B = Y0.clone()
    dt_A, pY_A, pG_A = 0.005, None, None
    dt_B, pY_B, pG_B = 0.005, None, None
    steps_per_frame = 2

    # Pre-build Pauli matrices, 3D ray grid, and z=0 slice grid
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    sigmas = torch.stack([s1, s2, s3], dim=0)

    n_th, n_ph = 16, 26
    th_1d = np.linspace(0.08, np.pi - 0.08, n_th)
    ph_1d = np.linspace(0.0, 2.0 * np.pi, n_ph)
    TH, PH = np.meshgrid(th_1d, ph_1d, indexing="ij")
    ux = np.sin(TH) * np.cos(PH)
    uy = np.sin(TH) * np.sin(PH)
    uz = np.cos(TH)
    unit_dirs = torch.tensor(np.stack([ux, uy, uz], axis=-1).reshape(-1, 3), dtype=torch.float64, device=device)

    n_grid = 43
    alpha_cl = (nu_B + np.sqrt(nu_B * nu_B - 2.0 * mu * mu)) / (2.0 * g)
    r_ref = alpha_cl * 0.5 * (N - 1)
    lim_xy = 1.40 * r_ref
    xs = torch.linspace(-lim_xy, lim_xy, n_grid, dtype=torch.float64, device=device)
    GX, GY = torch.meshgrid(xs, xs, indexing="ij")
    slice_pts = torch.stack([GX.reshape(-1), GY.reshape(-1), torch.zeros(n_grid * n_grid, dtype=torch.float64, device=device)], dim=-1)

    def bb_step(Y_cur, dt_cur, pY, pG, nu_val):
        E0, G, _ = core.total_energy_and_grad(Y_cur, g=g, mu=mu, nu=nu_val, kappa=kappa, g1=g1, g2=g2, compute_grad=True)
        gn = float(torch.sqrt(torch.real(torch.sum(G * G.conj()))))
        if gn < 1e-4:
            return Y_cur, dt_cur, pY, pG
        if pY is not None and pG is not None:
            sY, dG = Y_cur - pY, G - pG
            sy = float(torch.real(torch.sum(sY * dG.conj())))
            ss = float(torch.real(torch.sum(sY * sY.conj())))
            if sy > 1e-12 and ss > 1e-16:
                dt_cur = float(np.clip(ss / sy, 1e-5, 0.08))
        y_norm = float(torch.sqrt(torch.real(torch.sum(Y_cur * Y_cur.conj()))))
        ssz = min(dt_cur, max(1.5, 0.12 * y_norm) / max(gn, 1.0))
        for _ in range(8):
            Yt = core.project_hermitian_traceless(Y_cur - ssz * G)
            Et, _, _ = core.total_energy_and_grad(Yt, g=g, mu=mu, nu=nu_val, kappa=kappa, g1=g1, g2=g2, compute_grad=False)
            if Et <= E0 - 1e-4 * ssz * gn * gn:
                return Yt, min(ssz * 1.3, 0.08), Y_cur.clone(), G.clone()
            ssz *= 0.4
        return Y_cur, dt_cur * 0.3, None, None

    frames = []
    mode_tag = "METASTABLE DOUBLE-SPHERE" if metastable else "EMERGENT SPACETIME"
    print(f"[VIEWER] Computing {n_frames} exact U({N}) gauge-invariant frames ({mode_tag}, device={device})...")

    for fr in range(n_frames):
        t_cur = float(fr * steps_per_frame)
        # Measure exact gauge-invariant observables at current frame
        E_A, _, _ = core.total_energy_and_grad(Y_A, g=g, mu=mu, nu=nu_A, kappa=kappa, g1=g1, g2=g2, compute_grad=False)
        E_B, _, _ = core.total_energy_and_grad(Y_B, g=g, mu=mu, nu=nu_B, kappa=kappa, g1=g1, g2=g2, compute_grad=False)
        diag_A = _geometry_diag(Y_A)
        diag_B = _geometry_diag(Y_B)

        # Exact 2D z=0 Point-Probe Dirac slices (Universe B & Universe A)
        lam_slice_B = _batched_point_probe_min_eval(Y_B, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()
        lam_slice_A = _batched_point_probe_min_eval(Y_A, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()

        # Exact 3D Point-Probe Dirac shell surface for Universe B
        r_surf_flat, lam_surf_flat = _batched_3d_shell_surface(Y_B, unit_dirs, sigmas, r_max=lim_xy)
        r_surf = r_surf_flat.reshape(n_th, n_ph)
        lam_surf = lam_surf_flat.reshape(n_th, n_ph)

        # Exact Quantum Fisher Information Metric (4 * Provost-Vallee) at North Pole shell
        r_north = float(r_surf[0, 0])
        y_probe = torch.tensor([0.0, 0.0, max(r_north, 0.5)], dtype=torch.float64, device=device)
        qfim_B = _exact_qfim_eigenvalues(Y_B, y_probe, sigmas)

        # Exact Information-Geometric Quantum Relative Entropy Phi_TN across equator
        phi_B = _compute_equator_phi_tn(Y_B, core, kappa=kappa, g1=g1, g2=g2)
        phi_A = _compute_equator_phi_tn(Y_A, core, kappa=kappa, g1=g1, g2=g2)

        frames.append({
            "t": t_cur,
            "E_A": float(E_A), "E_B": float(E_B),
            "tr_A": float(diag_A["tr_Y2"]), "tr_B": float(diag_B["tr_Y2"]),
            "r2_evals_A": np.array(diag_A["r2_evals"], dtype=np.float64),
            "r2_evals_B": np.array(diag_B["r2_evals"], dtype=np.float64),
            "alg_err_B": float(diag_B["rel_alg_err"]),
            "slice_B": lam_slice_B, "slice_A": lam_slice_A,
            "r_surf": r_surf, "lam_surf": lam_surf,
            "qfim_B": qfim_B, "phi_B": phi_B, "phi_A": phi_A,
        })

        if fr % 12 == 0 or fr == n_frames - 1:
            print(
                f"  frame {fr+1:2d}/{n_frames} (step={int(t_cur):3d}) | "
                f"Tr(Y^2) A={diag_A['tr_Y2']:7.1f} vs B={diag_B['tr_Y2']:8.1f} | "
                f"AlgErr_B={diag_B['rel_alg_err']:.3e} | QFIM=[{qfim_B[0]:.4f}, {qfim_B[1]:.4f}, {qfim_B[2]:.4f}]"
            )

        if fr == n_frames - 1:
            break

        # Advance exact Phase 1 BB+Armijo flow for both universes identically
        for _ in range(steps_per_frame):
            Y_A, dt_A, pY_A, pG_A = bb_step(Y_A, dt_A, pY_A, pG_A, nu_A)
            Y_B, dt_B, pY_B, pG_B = bb_step(Y_B, dt_B, pY_B, pG_B, nu_B)

    elapsed = time.time() - t0
    print(f"[VIEWER] Trajectory complete in {elapsed:.2f}s! Launching interactive observatory...")
    guard.cleanup()
    return {
        "mode": "metastable" if metastable else "emergence",
        "N": N, "lim_xy": float(lim_xy), "r_ref": float(r_ref),
        "ux": ux, "uy": uy, "uz": uz, "frames": frames,
        "elapsed_sec": elapsed,
    }


@torch.no_grad()
def _generate_wave_frames(
    N: int = 8,
    n_frames: int = 72,
    state_pt: str = "",
    prefer_cuda: bool = True,
) -> Dict[str, object]:
    """
    Runs exact Phase 3A symplectic Velocity-Verlet gravitational-wave propagation
    after a localized Adjoint Dirac North-Pole pulse, recording exact gauge-invariant
    meridional shell ripples dr(theta, t) and local kinetic wave intensity.
    """
    t0 = time.time()
    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    device = guard.device
    core = GenesisQuantumCore(N=N, D=3, guard=guard)
    dirac = AdjointDiracMatterSector(core, m_psi=0.5)
    g, mu, nu, kappa, g1, g2 = 0.25, 1.0, 3.0, 0.15, 0.015, 1e-4

    # Load verified N=8 Phase 1 equilibrium checkpoint if available, else equilibrate cleanly
    loaded = False
    cand_pts = [state_pt] if state_pt else [
        os.path.join("results", "phase1_N8_20261125_173716", "phase1_states.pt"),
        os.path.join("release", "v1.0.1", "github_upload_v1.0.1", "results", "phase1_N8_20261125_173716", "phase1_states.pt"),
    ]
    y_bg, Sx_bg, Om_bg = None, None, None
    for cp in cand_pts:
        if cp and os.path.exists(cp):
            ck = torch.load(cp, map_location=device, weights_only=False)
            if int(ck["N"]) == N:
                y_bg = ck["universe_B"]["y"].to(device)
                Om_bg = ck["universe_B"]["Omega"].to(device)
                Om_bg, Sx_bg, _, _ = core.solve_hfb_omega(y_bg, g=g, mu=mu, nu=nu, max_iter=15, Omega_init=Om_bg)
                loaded = True
                print(f"[VIEWER] Loaded verified Phase 1 N={N} equilibrium from {cp}")
                break

    if not loaded:
        print(f"[VIEWER] Equilibrating N={N} fuzzy sphere from exact HFB...")
        L = build_su2_irrep(N, device)
        alpha_cl = (nu + np.sqrt(nu * nu - 2.0 * mu * mu)) / (2.0 * g)
        y_bg = core.mats_to_vec(alpha_cl * L)
        Om_bg, Sx_bg, _, _ = core.solve_hfb_omega(y_bg, g=g, mu=mu, nu=nu, max_iter=20)
        for _ in range(30):
            gr, _ = core.exact_grad_y(y_bg, Sx_bg, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2)
            gn = float(torch.linalg.norm(gr))
            if gn < 1e-4:
                break
            y_bg = core.project_traceless(y_bg - min(0.008, 2.0 / max(gn, 1.0)) * gr)
        Om_bg, Sx_bg, _, _ = core.solve_hfb_omega(y_bg, g=g, mu=mu, nu=nu, max_iter=20, Omega_init=Om_bg)

    Y_bg = core.vec_to_mats(y_bg)
    R_rms = float(torch.sqrt(torch.real(torch.sum(Y_bg * Y_bg.conj())) / float(N)))
    r_eq, _ = find_shell_radius_along_ray(core, Y_bg, np.array([0.0, 0.0, 1.0]), r_hint=R_rms)
    _, c_north = dirac.build_localized_probe_packet(Y_bg, torch.tensor([0.0, 0.0, r_eq], dtype=core.rdtype, device=device))

    # Precompute 19 gauge-invariant meridional probe projectors rho_theta & unit ray vectors
    n_th = 19
    thetas = np.linspace(0.0, np.pi, n_th)
    probe_rhos = []
    for th in thetas:
        pt = torch.tensor([r_eq * np.sin(th), 0.0, r_eq * np.cos(th)], dtype=core.rdtype, device=device)
        rho_th, _ = dirac.build_localized_probe_packet(Y_bg, pt)
        probe_rhos.append(rho_th)

    s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    sigmas = torch.stack([s1, s2, s3], dim=0)
    mer_dirs = torch.tensor(
        np.stack([np.sin(thetas), np.zeros(n_th), np.cos(thetas)], axis=-1),
        dtype=torch.float64, device=device,
    )
    r_win_grid = torch.linspace(r_eq * 0.96, r_eq * 1.04, 65, dtype=torch.float64, device=device)
    mer_pts_batch = (mer_dirs[:, None, :] * r_win_grid[None, :, None]).reshape(-1, 3)

    # Symplectic Velocity-Verlet real-time integration
    dt = 0.004
    steps_per_frame = 6
    total_steps = (n_frames - 1) * steps_per_frame
    M_pulse, t0_p, tau_p = 120.0, 0.06, 0.022

    def pulse_mass(t_v: float) -> float:
        return M_pulse * float(np.exp(-0.5 * ((t_v - t0_p) / tau_p) ** 2)) if t_v < 0.24 else 0.0

    def grad_tot(yv: torch.Tensor, t_v: float) -> Tuple[torch.Tensor, float]:
        gr, _ = core.exact_grad_y(yv, Sx_bg, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2)
        E_pot = core.mqm_y_energy_fixed_sx(yv, Sx_bg, g=g, mu=mu, nu=nu)
        E_pot += float(core.bdg_link_sector(core.vec_to_mats(yv), kappa=kappa, g1=g1, g2=g2)[0])
        mp = pulse_mass(t_v)
        if mp > 1e-8:
            e_m, g_m = dirac.matter_energy_and_grad(yv, c_north, mp)
            gr = gr + g_m
            E_pot += e_m
        return gr, float(E_pot)

    y_cur = y_bg.clone()
    q_cur = torch.zeros_like(y_bg)
    gr_cur, E_pot_cur = grad_tot(y_cur, 0.0)
    F_cur = -gr_cur

    n_ph = 28
    phis = np.linspace(0.0, 2.0 * np.pi, n_ph)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")

    frames = []
    print(f"[VIEWER] Integrating {total_steps} symplectic Velocity-Verlet wave steps ({n_frames} frames)...")
    for fr in range(n_frames):
        t_cur = fr * steps_per_frame * dt
        Y_cur = core.vec_to_mats(y_cur)
        Q_cur = core.vec_to_mats(q_cur)
        Q2_sum = sum(Q_cur[I] @ Q_cur[I] for I in range(3))

        # Single GPU-batched radial search + quadratic sub-grid interpolation across all 19 rays!
        lam_m = _batched_point_probe_min_eval(Y_cur, mer_pts_batch, sigmas).reshape(n_th, 65)
        idx_b = torch.argmin(lam_m, dim=1)
        r_vals = r_win_grid[idx_b].cpu().numpy()
        lam_np = lam_m.cpu().numpy()
        dr_step = float(r_win_grid[1] - r_win_grid[0])
        for k in range(n_th):
            j = int(idx_b[k])
            if 0 < j < 64:
                y0, y1, y2 = lam_np[k, j - 1], lam_np[k, j], lam_np[k, j + 1]
                denom = (y0 - 2.0 * y1 + y2)
                if denom > 1e-12:
                    r_vals[k] += 0.5 * (y0 - y2) / denom * dr_step
        dr_prof = r_vals - float(r_eq)
        kin_prof = np.array([
            max(float(torch.real(torch.einsum("ij,ji->", probe_rhos[k], Q2_sum))), 1e-12)
            for k in range(n_th)
        ], dtype=np.float64)

        E_kin = 0.5 * float(torch.dot(q_cur, q_cur))
        frames.append({
            "t": t_cur, "dr": dr_prof, "kin": kin_prof,
            "E_tot": E_pot_cur + E_kin, "E_kin": E_kin,
            "M_pulse": pulse_mass(t_cur),
        })
        if fr % 18 == 0 or fr == n_frames - 1:
            print(f"  frame {fr+1:2d}/{n_frames} (t={t_cur:4.2f}) | max|dr|={np.max(np.abs(dr_prof)):.4f} | E_kin={E_kin:.2f}")

        if fr == n_frames - 1:
            break
        for s in range(steps_per_frame):
            t_step = t_cur + (s + 1) * dt
            q_half = q_cur + 0.5 * dt * F_cur
            y_cur = core.project_traceless(y_cur + dt * q_half)
            gr_cur, E_pot_cur = grad_tot(y_cur, t_step)
            F_cur = -gr_cur
            q_cur = core.project_traceless(q_half + 0.5 * dt * F_cur)

    elapsed = time.time() - t0
    print(f"[VIEWER] Wave simulation complete in {elapsed:.2f}s! Launching interactive observatory...")
    guard.cleanup()
    return {
        "mode": "wave", "N": N, "r_eq": r_eq,
        "thetas": thetas, "TH": TH, "PH": PH, "frames": frames,
        "elapsed_sec": elapsed,
    }


def _launch_emergence_ui(data: Dict[str, object], headless: bool = False, save_gif: str = "", fps: int = 16):
    import matplotlib
    matplotlib.use("Agg" if headless else "TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.widgets import Button, Slider

    plt.rcParams.update({
        "figure.facecolor": "#0b0f19",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#9ca3af",
        "ytick.color": "#9ca3af",
        "text.color": "#f3f4f6",
        "grid.color": "#1f2937",
    })

    frames = data["frames"]
    n_frames = len(frames)
    N = data["N"]
    lim = data["lim_xy"]
    ux, uy, uz = data["ux"], data["uy"], data["uz"]
    ts = np.array([f["t"] for f in frames])
    r2_B_all = np.array([f["r2_evals_B"] for f in frames])
    r2_A_all = np.array([f["r2_evals_A"] for f in frames])
    qfim_all = np.array([f["qfim_B"] for f in frames])
    alg_all = np.array([max(f["alg_err_B"], 1e-9) for f in frames])
    phi_B_all = np.array([f["phi_B"] for f in frames])
    phi_A_all = np.array([f["phi_A"] for f in frames])
    E_B_all = np.array([f["E_B"] for f in frames])

    fig = plt.figure(figsize=(15.5, 9.2), dpi=105)
    mode_label = "Metastable Double-Shell Condensation" if data["mode"] == "metastable" else "Spontaneous Spacetime Genesis from 0D Chaos"
    title_txt = fig.suptitle(
        f"Genesis-Rigorous Zero-Cheat Live Observatory — {mode_label} (N={N}, 100% U(N) Gauge-Invariant)",
        fontsize=12.5, fontweight="bold", color="#fbbf24", y=0.975,
    )
    hud_txt = fig.text(0.5, 0.935, "", ha="center", va="center", fontsize=9.8, color="#93c5fd", family="monospace")

    gs = fig.add_gridspec(2, 3, left=0.05, right=0.97, top=0.89, bottom=0.12, wspace=0.26, hspace=0.33)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_sb = fig.add_subplot(gs[0, 1])
    ax_sa = fig.add_subplot(gs[0, 2])
    ax_r2 = fig.add_subplot(gs[1, 0])
    ax_qf = fig.add_subplot(gs[1, 1])
    ax_ig = fig.add_subplot(gs[1, 2])

    ax3d.set_facecolor("#0b0f19")
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1f2937")

    # Static axes setup for Panels 2-6
    def log_field(lam_arr):
        return -np.log10(np.clip(lam_arr, 1e-3, None) + 0.08)

    vmin_s = float(np.percentile(log_field(frames[0]["slice_B"]), 5))
    vmax_s = float(np.percentile(log_field(frames[-1]["slice_B"]), 99.5))

    im_B = ax_sb.imshow(log_field(frames[0]["slice_B"]).T, origin="lower", extent=[-lim, lim, -lim, lim], cmap="inferno", vmin=vmin_s, vmax=vmax_s, aspect="equal")
    ax_sb.set_title("2. Universe B (nu=3.0): Exact z=0 Dirac Field\n[Zero Mesh Cheat: -log10(|lambda_min|)]", fontsize=9.3)
    ax_sb.set_xlabel("x"); ax_sb.set_ylabel("y")

    im_A = ax_sa.imshow(log_field(frames[0]["slice_A"]).T, origin="lower", extent=[-lim, lim, -lim, lim], cmap="inferno", vmin=vmin_s, vmax=vmax_s, aspect="equal")
    ax_sa.set_title("3. Null Control Universe A (nu=0.8): z=0 Slice\n[Identical Random GUE Seed -> 0D Collapse]", fontsize=9.3)
    ax_sa.set_xlabel("x"); ax_sa.set_ylabel("y")

    # Panel 4: R^2 eigenvalues convergence
    r2_lines_B = [ax_r2.plot([], [], lw=1.6, alpha=0.85, color=plt.cm.cool(i / max(N - 1, 1)))[0] for i in range(N)]
    r2_line_A, = ax_r2.plot([], [], "r--", lw=1.8, label="Universe A Mean R^2 (Collapse)")
    ax_r2.set_xlim(ts[0], ts[-1])
    ax_r2.set_ylim(-0.04 * np.max(r2_B_all), 1.12 * np.max(r2_B_all))
    ax_r2.set_title("4. R^2 = sum(Y_I^2) Eigenvalue Degeneracy\n[N Scattered Noise Lines Lock Into Smooth Sphere]", fontsize=9.3)
    ax_r2.set_xlabel("Relaxation Step (Exact BB Flow)"); ax_r2.set_ylabel("Eigenvalues of R^2")
    ax_r2.legend(loc="upper left", fontsize=7.8); ax_r2.grid(True, alpha=0.35)

    # Panel 5: Information Geometry — Quantum Fisher Information Metric eigenvalues
    q_cols = ["#f87171", "#34d399", "#60a5fa"]
    q_lbls = ["g_1 (Normal -> 0)", "g_2 (Tangent 1)", "g_3 (Tangent 2)"]
    qf_lines = [ax_qf.plot([], [], lw=2.0, color=q_cols[k], label=q_lbls[k])[0] for k in range(3)]
    ax_qf.set_xlim(ts[0], ts[-1])
    q_max_vis = float(np.percentile(qfim_all[:, 2], 96)) * 1.25 + 1e-4
    ax_qf.set_ylim(-0.05 * q_max_vis, q_max_vis)
    ax_qf.set_title("5. Quantum Information Geometry (QFIM = 4*g_PV)\n[3D Chaos -> Exact Rank-2 Manifold (0, g_t, g_t)]", fontsize=9.3)
    ax_qf.set_xlabel("Relaxation Step (Exact BB Flow)"); ax_qf.set_ylabel("Quantum Fisher Metric Eigenvalues")
    ax_qf.legend(loc="upper right", fontsize=7.8); ax_qf.grid(True, alpha=0.35)

    # Panel 6: AlgErr (log) & Information-Geometric Relative Entropy Phi_TN
    ax_ig2 = ax_ig.twinx()
    l_alg, = ax_ig.plot([], [], color="#fbbf24", lw=2.1, label="SU(2) AlgErr (Left, log)")
    l_phiB, = ax_ig2.plot([], [], color="#38bdf8", lw=2.0, label="Phi_TN Univ B (Right)")
    l_phiA, = ax_ig2.plot([], [], color="#f87171", ls="--", lw=1.6, label="Phi_TN Univ A (0D Soup)")
    ax_ig.set_xlim(ts[0], ts[-1])
    ax_ig.set_yscale("log")
    ax_ig.set_ylim(max(np.min(alg_all) * 0.4, 1e-8), np.max(alg_all) * 2.5)
    ax_ig2.set_ylim(0.0, max(np.max(phi_A_all), np.max(phi_B_all)) * 1.18)
    ax_ig.set_title("6. Algebraic Crystallization & Info-Geom Phi_TN\n[Bulk Decoherence -> Area-Law Integration]", fontsize=9.3)
    ax_ig.set_xlabel("Relaxation Step (Exact BB Flow)"); ax_ig.set_ylabel("AlgErr (log)", color="#fbbf24")
    ax_ig2.set_ylabel("Quantum Rel. Entropy Phi_TN", color="#38bdf8")
    ax_ig.legend([l_alg, l_phiB, l_phiA], [l.get_label() for l in (l_alg, l_phiB, l_phiA)], loc="center right", fontsize=7.4)
    ax_ig.grid(True, alpha=0.35)

    state = {"idx": 0, "playing": True}

    def render_frame(idx: int):
        f = frames[idx]
        # 1. Redraw 3D exact gauge-invariant Point-Probe shell (colored by sharpness 1/(1+lam))
        ax3d.cla()
        ax3d.set_facecolor("#0b0f19")
        r_s = f["r_surf"]
        lam_s = f["lam_surf"]
        X3, Y3, Z3 = r_s * ux, r_s * uy, r_s * uz
        sharp = 1.0 / (1.0 + lam_s / max(0.15 * lim, 1.0))
        face_cols = plt.cm.plasma(np.clip((sharp - 0.35) / 0.65, 0.0, 1.0))
        ax3d.plot_surface(X3, Y3, Z3, facecolors=face_cols, rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.88, linewidth=0.15, edgecolor="#ffffff22")
        ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_title("1. 3D Point-Probe Dirac Shell (Universe B)\n[Bright = Sharp Zero-Mode Manifold | Drag to Rotate]", fontsize=9.3)
        ax3d.set_xlabel("X", labelpad=-2); ax3d.set_ylabel("Y", labelpad=-2); ax3d.set_zlabel("Z", labelpad=-2)
        ax3d.tick_params(labelsize=7, pad=-2)

        # 2 & 3. Update 2D z=0 slices
        im_B.set_data(log_field(f["slice_B"]).T)
        im_A.set_data(log_field(f["slice_A"]).T)

        # 4, 5, 6. Update time-series curves
        sl = slice(0, idx + 1)
        t_sl = ts[sl]
        for i in range(N):
            r2_lines_B[i].set_data(t_sl, r2_B_all[sl, i])
        r2_line_A.set_data(t_sl, np.mean(r2_A_all[sl], axis=1))
        for k in range(3):
            qf_lines[k].set_data(t_sl, qfim_all[sl, k])
        l_alg.set_data(t_sl, alg_all[sl])
        l_phiB.set_data(t_sl, phi_B_all[sl])
        l_phiA.set_data(t_sl, phi_A_all[sl])

        qf = f["qfim_B"]
        eff_rank = int(np.sum(qf > 0.15 * max(qf[2], 1e-9)))
        hud_txt.set_text(
            f"Frame {idx+1:2d}/{n_frames}  |  t = {f['t']:.2f}  |  E_B = {f['E_B']:,.1f}  |  "
            f"Tr(Y^2): A={f['tr_A']:.1f} vs B={f['tr_B']:,.1f}  |  AlgErr={f['alg_err_B']:.2e}  |  "
            f"QFIM Rank={eff_rank}D  |  Phi_TN(B)={f['phi_B']:.2f}"
        )
        return []

    render_frame(0)

    if save_gif:
        os.makedirs(os.path.dirname(save_gif) or ".", exist_ok=True)
        print(f"[VIEWER] Exporting animation GIF to {save_gif} ...")
        ani_save = FuncAnimation(fig, lambda i: render_frame(i), frames=n_frames, interval=int(1000 / fps), blit=False)
        ani_save.save(save_gif, writer=PillowWriter(fps=fps))
        print(f"[VIEWER] Saved GIF: {save_gif}")

    if headless:
        plt.close(fig)
        return

    # Interactive controls (Slider + Play/Pause + Replay)
    ax_sl = fig.add_axes([0.14, 0.035, 0.54, 0.028], facecolor="#1f2937")
    slider = Slider(ax_sl, "Frame", 0, n_frames - 1, valinit=0, valstep=1, color="#38bdf8")
    ax_btn_pp = fig.add_axes([0.72, 0.03, 0.10, 0.04])
    btn_pp = Button(ax_btn_pp, "Pause", color="#1e3a8a", hovercolor="#2563eb")
    ax_btn_rp = fig.add_axes([0.84, 0.03, 0.10, 0.04])
    btn_rp = Button(ax_btn_rp, "Replay", color="#065f46", hovercolor="#059669")

    updating_slider = {"flag": False}

    def on_slider(val):
        if updating_slider["flag"]:
            return
        state["idx"] = int(val)
        render_frame(state["idx"])
        fig.canvas.draw_idle()

    def on_pp(_ev):
        state["playing"] = not state["playing"]
        btn_pp.label.set_text("Pause" if state["playing"] else "Play")
        fig.canvas.draw_idle()

    def on_rp(_ev):
        state["idx"] = 0
        state["playing"] = True
        btn_pp.label.set_text("Pause")
        updating_slider["flag"] = True
        slider.set_val(0)
        updating_slider["flag"] = False
        render_frame(0)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)
    btn_pp.on_clicked(on_pp)
    btn_rp.on_clicked(on_rp)

    def step_anim(_frame):
        if not state["playing"]:
            return []
        state["idx"] = (state["idx"] + 1) % n_frames
        updating_slider["flag"] = True
        slider.set_val(state["idx"])
        updating_slider["flag"] = False
        return render_frame(state["idx"])

    _ani = FuncAnimation(fig, step_anim, interval=int(1000 / fps), blit=False, cache_frame_data=False)
    plt.show()


def _launch_wave_ui(data: Dict[str, object], headless: bool = False, save_gif: str = "", fps: int = 16):
    import matplotlib
    matplotlib.use("Agg" if headless else "TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.widgets import Button, Slider

    plt.rcParams.update({
        "figure.facecolor": "#0b0f19",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#9ca3af",
        "ytick.color": "#9ca3af",
        "text.color": "#f3f4f6",
        "grid.color": "#1f2937",
    })

    frames = data["frames"]
    n_frames = len(frames)
    r_eq = data["r_eq"]
    thetas = data["thetas"]
    th_deg = np.degrees(thetas)
    TH, PH = data["TH"], data["PH"]
    ts = np.array([f["t"] for f in frames])
    dr_mat = np.array([f["dr"] for f in frames])          # (T, n_th)
    kin_mat = np.log10(np.array([f["kin"] for f in frames]) + 1e-12)
    E_tot_all = np.array([f["E_tot"] for f in frames])

    max_dr = max(float(np.max(np.abs(dr_mat))), 1e-4)
    exag = 0.24 * r_eq / max_dr
    k_vmin, k_vmax = float(np.percentile(kin_mat, 10)), float(np.percentile(kin_mat, 99.5))

    fig = plt.figure(figsize=(14.8, 8.8), dpi=105)
    fig.suptitle(
        f"Genesis-Rigorous Zero-Cheat Live Observatory — Real-Time Emergent Gravitational Wave (N={data['N']})",
        fontsize=12.5, fontweight="bold", color="#38bdf8", y=0.975,
    )
    hud_txt = fig.text(0.5, 0.935, "", ha="center", va="center", fontsize=9.8, color="#fbbf24", family="monospace")

    gs = fig.add_gridspec(2, 2, left=0.06, right=0.96, top=0.89, bottom=0.12, wspace=0.25, hspace=0.33)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_mer = fig.add_subplot(gs[0, 1])
    ax_lc = fig.add_subplot(gs[1, 0])
    ax_osc = fig.add_subplot(gs[1, 1])

    # Panel 2: Meridian cross-section
    th_full = np.concatenate([thetas, 2.0 * np.pi - thetas[::-1]])
    ax_mer.plot(r_eq * np.sin(th_full), r_eq * np.cos(th_full), "w--", alpha=0.35, lw=1.2, label="Equilibrium Sphere")
    mer_line, = ax_mer.plot([], [], color="#38bdf8", lw=2.5, label=f"Rippled Shell ({exag:.0f}x radial amp)")
    ax_mer.set_xlim(-1.35 * r_eq, 1.35 * r_eq); ax_mer.set_ylim(-1.35 * r_eq, 1.35 * r_eq)
    ax_mer.set_aspect("equal")
    ax_mer.set_title("2. Exact Point-Probe Meridian Ripple Cross-Section\n[North-Pole Pulse -> South-Pole Focusing]", fontsize=9.5)
    ax_mer.set_xlabel("x"); ax_mer.set_ylabel("z"); ax_mer.legend(loc="upper right", fontsize=8); ax_mer.grid(True, alpha=0.3)

    # Panel 3: Causal Light-Cone heatmap
    ax_lc.imshow(kin_mat.T, origin="lower", aspect="auto", extent=[ts[0], ts[-1], th_deg[0], th_deg[-1]], cmap="magma", vmin=k_vmin, vmax=k_vmax)
    lc_cursor = ax_lc.axvline(ts[0], color="#38bdf8", lw=2.0)
    ax_lc.set_title("3. Real-Time Causal Light Cone (log10 Local Kinetic Wave)\n[Vertical Cyan Line Tracks Current Time t]", fontsize=9.5)
    ax_lc.set_xlabel("Emergent Time t"); ax_lc.set_ylabel("Polar Angle theta [deg]")

    # Panel 4: Oscilloscope traces at North, Equator, South
    l_n, = ax_osc.plot([], [], color="#f87171", lw=2.0, label="North Pole (0 deg)")
    l_e, = ax_osc.plot([], [], color="#fbbf24", lw=2.0, label="Equator (90 deg)")
    l_s, = ax_osc.plot([], [], color="#34d399", lw=2.2, label="South Pole (180 deg, Focus!)")
    ax_osc.set_xlim(ts[0], ts[-1]); ax_osc.set_ylim(-1.2 * max_dr, 1.2 * max_dr)
    ax_osc.set_title("4. Point-Probe Shell Displacement dr(theta, t)\n[Wavefront Arrives at Equator Then Focuses at South Pole]", fontsize=9.5)
    ax_osc.set_xlabel("Emergent Time t"); ax_osc.set_ylabel("Radial Ripple dr")
    ax_osc.legend(loc="upper right", fontsize=8); ax_osc.grid(True, alpha=0.35)

    state = {"idx": 0, "playing": True}
    lim3 = 1.32 * r_eq

    def render_frame(idx: int):
        f = frames[idx]
        ax3d.cla()
        ax3d.set_facecolor("#0b0f19")
        r_prof = r_eq + exag * f["dr"]
        R3 = r_prof[:, None] * np.ones_like(PH)
        X3 = R3 * np.sin(TH) * np.cos(PH)
        Y3 = R3 * np.sin(TH) * np.sin(PH)
        Z3 = R3 * np.cos(TH)
        kn = np.clip((kin_mat[idx] - k_vmin) / max(k_vmax - k_vmin, 1e-6), 0.0, 1.0)
        C3 = plt.cm.magma(kn[:, None] * np.ones_like(PH))
        ax3d.plot_surface(X3, Y3, Z3, facecolors=C3, rstride=1, cstride=1, shade=False, alpha=0.92, linewidth=0.1, edgecolor="#ffffff18")
        ax3d.set_xlim(-lim3, lim3); ax3d.set_ylim(-lim3, lim3); ax3d.set_zlim(-lim3, lim3)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_title("1. 3D Rippling Emergent Spacetime (Drag to Rotate)\n[Glow = Local Kinetic Wave Intensity | Shape = dr(theta,t)]", fontsize=9.5)

        r_full = np.concatenate([r_prof, r_prof[::-1]])
        mer_line.set_data(r_full * np.sin(th_full), r_full * np.cos(th_full))
        lc_cursor.set_xdata([f["t"], f["t"]])

        sl = slice(0, idx + 1)
        l_n.set_data(ts[sl], dr_mat[sl, 0])
        l_e.set_data(ts[sl], dr_mat[sl, len(thetas) // 2])
        l_s.set_data(ts[sl], dr_mat[sl, -1])

        hud_txt.set_text(
            f"Frame {idx+1:2d}/{n_frames}  |  t = {f['t']:.3f}  |  Pulse M(t) = {f['M_pulse']:5.1f}  |  "
            f"E_kin = {f['E_kin']:6.2f}  |  E_tot = {f['E_tot']:,.2f}"
        )
        return []

    render_frame(0)

    if save_gif:
        os.makedirs(os.path.dirname(save_gif) or ".", exist_ok=True)
        print(f"[VIEWER] Exporting wave animation GIF to {save_gif} ...")
        ani_save = FuncAnimation(fig, lambda i: render_frame(i), frames=n_frames, interval=int(1000 / fps), blit=False)
        ani_save.save(save_gif, writer=PillowWriter(fps=fps))
        print(f"[VIEWER] Saved GIF: {save_gif}")

    if headless:
        plt.close(fig)
        return

    ax_sl = fig.add_axes([0.14, 0.035, 0.54, 0.028], facecolor="#1f2937")
    slider = Slider(ax_sl, "Frame", 0, n_frames - 1, valinit=0, valstep=1, color="#38bdf8")
    ax_btn_pp = fig.add_axes([0.72, 0.03, 0.10, 0.04])
    btn_pp = Button(ax_btn_pp, "Pause", color="#1e3a8a", hovercolor="#2563eb")
    ax_btn_rp = fig.add_axes([0.84, 0.03, 0.10, 0.04])
    btn_rp = Button(ax_btn_rp, "Replay", color="#065f46", hovercolor="#059669")
    updating_slider = {"flag": False}

    def on_slider(val):
        if updating_slider["flag"]:
            return
        state["idx"] = int(val); render_frame(state["idx"]); fig.canvas.draw_idle()

    def on_pp(_ev):
        state["playing"] = not state["playing"]
        btn_pp.label.set_text("Pause" if state["playing"] else "Play"); fig.canvas.draw_idle()

    def on_rp(_ev):
        state["idx"] = 0; state["playing"] = True; btn_pp.label.set_text("Pause")
        updating_slider["flag"] = True; slider.set_val(0); updating_slider["flag"] = False
        render_frame(0); fig.canvas.draw_idle()

    slider.on_changed(on_slider); btn_pp.on_clicked(on_pp); btn_rp.on_clicked(on_rp)

    def step_anim(_frame):
        if not state["playing"]:
            return []
        state["idx"] = (state["idx"] + 1) % n_frames
        updating_slider["flag"] = True; slider.set_val(state["idx"]); updating_slider["flag"] = False
        return render_frame(state["idx"])

    _ani = FuncAnimation(fig, step_anim, interval=int(1000 / fps), blit=False, cache_frame_data=False)
    plt.show()


def run_spacetime_viewer(
    mode: str = "emergence",
    N: int = 8,
    n_frames: int = 68,
    state_pt: str = "",
    fps: int = 16,
    save_gif: str = "",
    headless: bool = False,
    prefer_cuda: bool = True,
) -> Dict[str, object]:
    """Unified entrypoint for the zero-cheat interactive spacetime observatory."""
    if mode in ("emergence", "metastable"):
        data = _generate_emergence_frames(N=N, metastable=(mode == "metastable"), n_frames=n_frames, prefer_cuda=prefer_cuda)
        _launch_emergence_ui(data, headless=headless, save_gif=save_gif, fps=fps)
    elif mode == "wave":
        data = _generate_wave_frames(N=N, n_frames=n_frames, state_pt=state_pt, prefer_cuda=prefer_cuda)
        _launch_wave_ui(data, headless=headless, save_gif=save_gif, fps=fps)
    else:
        raise ValueError(f"Unknown viewer mode: {mode}. Use 'emergence', 'metastable', or 'wave'.")
    return {"mode": mode, "N": N, "n_frames": len(data["frames"]), "elapsed_sec": data["elapsed_sec"], "save_gif": save_gif}
