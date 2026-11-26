"""
Genesis-Rigorous v1.0.2 — Classic 5-Panel Zero-Cheat Spacetime & Quantum Melting Viewer.

Preserves the clean, iconic v1.0.1 visual aesthetic (dark observatory theme, static
non-rotating 3D Point-Probe shell, glowing inferno z=0 Dirac cross-section slices,
R^2 eigenvalue convergence/dissolution, and Quantum Fisher Information Metric QFIM),
while removing the v1.0.1 Phi_TN panel and adding exact U(N)-singlet HMC quantum melting.

Modes:
  --mode birth     : Exact U(N)-singlet gauged HMC (v1.0.2 Exp 2) starting from pure melted
                     0D quantum fog (Haar-random U, random X, |P|~0.2) and cooling g: 1.35 -> 0.25
                     so a 2D fuzzy sphere spontaneously nucleates and crystallizes out of quantum
                     chaos, side-by-side with an identical-seed sub-critical HMC control (nu=0.80).
  --mode melting   : Exact U(N)-singlet gauged HMC (v1.0.2 Exp 2) sweeping quantum coupling
                     g: 0.24 -> 1.30 so a crisp glowing fuzzy sphere physically shrinks,
                     fractures, and melts into a quantum matrix liquid (side-by-side with
                     the initial solid-sphere reference slice).
  --mode emergence : Spontaneous condensation from 0D isotropic GUE noise into a 2D fuzzy
                     sphere (Universe B, nu=3.0) via gradient-flow relaxation, side-by-side
                     with Universe A (nu=0.8) trivial-vacuum collapse.
"""

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch

from ..exact_hfb import build_su2_irrep
from ..hmc_singlet import EPS3_PAIRS, ExactSingletMQMHMC
from ..matrix_native import MatrixNativeQuantumCore
from ..safety import HardwareSafetyGuard, SafetyConfig, safe_eigh, safe_eigvalsh


@torch.no_grad()
def _batched_point_probe_min_eval(
    Y: torch.Tensor, pts: torch.Tensor, sigmas: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    N = Y.shape[1]
    device = Y.device
    cdtype = Y.dtype
    eye_N = torch.eye(N, dtype=cdtype, device=device)
    B = sum(torch.kron(Y[I].contiguous(), sigmas[I]) for I in range(3))
    B = 0.5 * (B + B.conj().T)
    S_kron = torch.stack([torch.kron(eye_N, sigmas[I]) for I in range(3)], dim=0)

    P = pts.shape[0]
    out = torch.empty(P, dtype=torch.float64, device=device)
    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        p_c = pts[start:end].to(cdtype)
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
    Y: torch.Tensor, unit_dirs: torch.Tensor, sigmas: torch.Tensor,
    r_max: float, n_coarse: int = 18, n_fine: int = 11
) -> Tuple[np.ndarray, np.ndarray]:
    device = Y.device
    R = unit_dirs.shape[0]
    r_grid = torch.linspace(0.0, r_max, n_coarse, dtype=torch.float64, device=device)
    pts_c = unit_dirs[:, None, :] * r_grid[None, :, None]
    lam_c = _batched_point_probe_min_eval(Y, pts_c.reshape(-1, 3), sigmas).reshape(R, n_coarse)
    idx_best = torch.argmin(lam_c, dim=1)
    r_best_c = r_grid[idx_best]
    dr = r_max / max(n_coarse - 1, 1)

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
    """Exact Quantum Fisher Information Metric g^QFI = 4 * g^PV sorted eigenvalues [g_1 <= g_2 <= g_3]."""
    N = Y.shape[1]
    device = Y.device
    cdtype = Y.dtype
    eye_N = torch.eye(N, dtype=cdtype, device=device)
    D_mat = sum(torch.kron((Y[I] - y_pt[I] * eye_N).contiguous(), sigmas[I]) for I in range(3))
    D_mat = 0.5 * (D_mat + D_mat.conj().T)
    evals, evecs = safe_eigh(D_mat)
    i0 = int(torch.argmin(torch.abs(evals)))
    lam0, psi0 = evals[i0], evecs[:, i0]
    mask = torch.arange(evals.shape[0], device=device) != i0
    d_lam = evals[mask] - lam0
    denom = torch.where(torch.abs(d_lam) > 1e-6, d_lam * d_lam, torch.full_like(d_lam, 1e12))
    V_oth = evecs[:, mask]
    M = torch.empty((V_oth.shape[1], 3), dtype=cdtype, device=device)
    for a in range(3):
        M[:, a] = V_oth.conj().T @ (torch.kron(eye_N, sigmas[a]) @ psi0)
    g_qfi = 4.0 * torch.real(torch.einsum("na,nb,n->ab", M.conj(), M, 1.0 / denom))
    g_qfi = 0.5 * (g_qfi + g_qfi.T)
    return np.sort(safe_eigvalsh(g_qfi).cpu().numpy())


@torch.no_grad()
def _geometry_diag(Y: torch.Tensor) -> Dict[str, object]:
    tr_y2 = float(torch.real(torch.sum(Y * Y.conj())))
    R2 = Y[0] @ Y[0] + Y[1] @ Y[1] + Y[2] @ Y[2]
    R2 = 0.5 * (R2 + R2.conj().T)
    r2_evals = np.sort(torch.real(safe_eigvalsh(R2)).cpu().numpy())
    if tr_y2 < 1e-10:
        return {"tr_Y2": tr_y2, "r2_evals": r2_evals, "rel_alg_err": 1.0}
    num_a, den_a = 0.0, 0.0
    for I, J, K, sgn in EPS3_PAIRS:
        C_IJ = -1j * (Y[I] @ Y[J] - Y[J] @ Y[I])
        num_a += float(torch.real(torch.trace(C_IJ @ (sgn * Y[K]))))
        den_a += float(torch.real(torch.trace(Y[K] @ Y[K])))
    a_fit = num_a / max(den_a, 1e-12)
    alg_res = 0.0
    for I, J, K, sgn in EPS3_PAIRS:
        C_IJ = -1j * (Y[I] @ Y[J] - Y[J] @ Y[I])
        diff = C_IJ - a_fit * sgn * Y[K]
        alg_res += float(torch.real(torch.sum(diff * diff.conj())))
    rel_err = float(np.sqrt(alg_res / max(2.0 * (a_fit ** 2) * tr_y2, 1e-12)))
    return {"tr_Y2": tr_y2, "r2_evals": r2_evals, "rel_alg_err": rel_err}


def _make_probe_grids(device: torch.device, lim_xy: float, n_th: int = 16, n_ph: int = 26, n_grid: int = 43):
    sigmas = torch.stack([
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
        torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
    ]).to(device)
    th_1d = np.linspace(0.08, np.pi - 0.08, n_th)
    ph_1d = np.linspace(0.0, 2.0 * np.pi, n_ph)
    TH, PH = np.meshgrid(th_1d, ph_1d, indexing="ij")
    ux, uy, uz = np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)
    unit_dirs = torch.tensor(np.stack([ux, uy, uz], axis=-1).reshape(-1, 3), dtype=torch.float64, device=device)

    xs = torch.linspace(-lim_xy, lim_xy, n_grid, dtype=torch.float64, device=device)
    GX, GY = torch.meshgrid(xs, xs, indexing="ij")
    slice_pts = torch.stack(
        [GX.reshape(-1), GY.reshape(-1), torch.zeros(n_grid * n_grid, dtype=torch.float64, device=device)], dim=-1
    )
    return sigmas, ux, uy, uz, unit_dirs, slice_pts, n_th, n_ph, n_grid


@torch.no_grad()
def _generate_birth_frames(N: int = 4, n_frames: int = 72, prefer_cuda: bool = True) -> Dict[str, object]:
    """
    Exact U(N)-Singlet Gauged HMC Quantum Birth / Nucleation trajectory (v1.0.2):
    Starts from a 100% melted 0D quantum liquid (random Gaussian X + Haar-random U, ZERO sphere seed)
    at g=1.35, and anneals g -> 0.25 at nu=2.30 (> nu_c^quantum ~ 1.65), side-by-side with a parallel
    exact U(N)-singlet HMC control universe at nu=0.80 (< nu_c) starting from the IDENTICAL quantum fog.
    """
    t0 = time.time()
    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    gpu = guard.device
    cpu = torch.device("cpu")
    L, beta, nu_live, nu_ctrl = 8, 2.5, 2.30, 0.80
    g_start, g_end = 1.35, 0.25

    hmc_live = ExactSingletMQMHMC(N=N, L=L, beta=beta, g=g_start, mu=1.0, nu=nu_live, gauged=True, device=cpu)
    hmc_ctrl = ExactSingletMQMHMC(N=N, L=L, beta=beta, g=g_start, mu=1.0, nu=nu_ctrl, gauged=True, device=cpu)
    gen_live = torch.Generator(device="cpu").manual_seed(44)
    gen_ctrl = torch.Generator(device="cpu").manual_seed(44)

    X_live = hmc_live.sample_hermitian_momentum((1, L, 3), gen_live) * 0.5
    Z = (
        torch.randn(N, N, dtype=torch.float64, generator=gen_live)
        + 1j * torch.randn(N, N, dtype=torch.float64, generator=gen_live)
    )
    Q, R = torch.linalg.qr(Z)
    U_live = (Q * (torch.diag(R) / torch.abs(torch.diag(R))).unsqueeze(0)).unsqueeze(0)
    dt_live, dt_ctrl = 0.04, 0.04

    # Thermalize completely in the melted 0D quantum liquid phase (g=1.35)
    for _ in range(30):
        X_live, U_live, acc, _ = hmc_live.hmc_step(X_live, U_live, dt_live, 6, gen_live)
    # Identical melted quantum-fog initial state for the sub-critical control universe!
    X_ctrl, U_ctrl = X_live.clone(), U_live.clone()

    a_end = (nu_live + np.sqrt(nu_live * nu_live - 2.0)) / (2.0 * g_end)
    r_ref = a_end * 0.5 * (N - 1)
    lim_xy = 1.45 * r_ref
    sigmas, ux, uy, uz, unit_dirs, slice_pts, n_th, n_ph, n_grid = _make_probe_grids(gpu, lim_xy)

    n_cool = min(56, max(n_frames - 12, int(0.78 * n_frames)))
    gs_cool = np.geomspace(g_start, g_end, n_cool)
    gs_sched = np.concatenate([gs_cool, np.full(max(n_frames - n_cool, 0), g_end)])[:n_frames]

    frames = []
    print(f"[VIEWER v1.0.2] Computing {n_frames} exact U({N})-singlet HMC BIRTH frames (g={g_start:.2f}->{g_end:.2f}, nu={nu_live} vs {nu_ctrl}, device={gpu})...")
    for fr in range(n_frames):
        g_cur = float(gs_sched[fr])
        hmc_live.g = g_cur
        hmc_ctrl.g = g_cur

        Y_live = X_live[0, 0].to(gpu)
        Y_ctrl = X_ctrl[0, 0].to(gpu)
        diag_live = _geometry_diag(Y_live)
        diag_ctrl = _geometry_diag(Y_ctrl)
        poly = float(torch.abs(torch.trace(U_live[0])) / float(N))

        slice_live = _batched_point_probe_min_eval(Y_live, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()
        slice_ctrl = _batched_point_probe_min_eval(Y_ctrl, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()
        r_surf_f, lam_surf_f = _batched_3d_shell_surface(Y_live, unit_dirs, sigmas, r_max=lim_xy)
        r_surf, lam_surf = r_surf_f.reshape(n_th, n_ph), lam_surf_f.reshape(n_th, n_ph)

        r_north = float(r_surf[0, 0])
        y_probe = torch.tensor([0.0, 0.0, max(r_north, 0.5)], dtype=torch.float64, device=gpu)
        qfim = _exact_qfim_eigenvalues(Y_live, y_probe, sigmas)
        qfim_norm = qfim / max(float(qfim[2]), 1e-12)

        frames.append({
            "x_val": float(fr),
            "g": g_cur, "nu": float(nu_live),
            "tr_live": float(diag_live["tr_Y2"]),
            "tr_ctrl": float(diag_ctrl["tr_Y2"]),
            "r2_evals": np.array(diag_live["r2_evals"], dtype=np.float64),
            "r2_ctrl": float(np.mean(diag_ctrl["r2_evals"])),
            "alg_err": float(diag_live["rel_alg_err"]),
            "poly": poly,
            "slice_main": slice_live,
            "slice_ctrl": slice_ctrl,
            "r_surf": r_surf, "lam_surf": lam_surf,
            "qfim": qfim, "qfim_norm": qfim_norm,
        })
        if fr % 12 == 0 or fr == n_frames - 1:
            print(
                f"  frame {fr+1:2d}/{n_frames} (g={g_cur:4.2f}) | Tr(X^2): Ctrl={diag_ctrl['tr_Y2']:5.1f} vs Live={diag_live['tr_Y2']:6.1f} | "
                f"AlgErr={diag_live['rel_alg_err']:.3f} | |P|={poly:.2f} | QFIM/g3=[{qfim_norm[0]:.3f}, {qfim_norm[1]:.3f}, {qfim_norm[2]:.3f}]"
            )
        if fr == n_frames - 1:
            break
        acc_l, acc_c = 0.0, 0.0
        for _ in range(10):
            X_live, U_live, a_l, _ = hmc_live.hmc_step(X_live, U_live, dt_live, 6, gen_live)
            X_ctrl, U_ctrl, a_c, _ = hmc_ctrl.hmc_step(X_ctrl, U_ctrl, dt_ctrl, 6, gen_ctrl)
            acc_l += float(a_l[0])
            acc_c += float(a_c[0])
        if acc_l / 10.0 < 0.50:
            dt_live = max(dt_live * 0.80, 0.002)
        elif acc_l / 10.0 > 0.85:
            dt_live = min(dt_live * 1.12, 0.08)
        if acc_c / 10.0 < 0.50:
            dt_ctrl = max(dt_ctrl * 0.80, 0.002)
        elif acc_c / 10.0 > 0.85:
            dt_ctrl = min(dt_ctrl * 1.12, 0.08)

    guard.cleanup()
    print(f"[VIEWER v1.0.2] Birth trajectory ready in {time.time()-t0:.1f}s!")
    return {
        "mode": "birth", "N": N, "lim_xy": float(lim_xy),
        "ux": ux, "uy": uy, "uz": uz, "frames": frames,
    }


@torch.no_grad()
def _generate_melting_frames(N: int = 6, n_frames: int = 72, prefer_cuda: bool = True) -> Dict[str, object]:
    """
    Exact U(N)-Singlet Gauged HMC melting trajectory (v1.0.2):
    Ramps quantum coupling g from 0.24 (solid fuzzy sphere) -> 1.30 (quantum matrix liquid)
    at fixed Myers flux nu = 2.30, recording exact 3D shells, z=0 inferno slices, R^2 spectra,
    Polyakov loop, and exact QFIM rank breakdown!
    """
    t0 = time.time()
    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    gpu = guard.device
    cpu = torch.device("cpu")
    L, beta, nu = 8, 2.5, 2.30
    g_start, g_end = 0.24, 1.30

    hmc = ExactSingletMQMHMC(N=N, L=L, beta=beta, g=g_start, mu=1.0, nu=nu, gauged=True, device=cpu)
    gen = torch.Generator(device="cpu").manual_seed(202612)
    a_init = (nu + np.sqrt(nu * nu - 2.0)) / (2.0 * g_start)
    Ls = build_su2_irrep(N, cpu)
    noise = hmc.sample_hermitian_momentum((1, L, 3), gen) * 0.10
    X = (float(a_init) * Ls).view(1, 1, 3, N, N).expand(1, L, 3, N, N) + noise
    U = torch.eye(N, dtype=torch.complex128, device=cpu).unsqueeze(0)
    dt = 0.020

    # Brief thermalization on solid fuzzy sphere
    for _ in range(30):
        X, U, acc, _ = hmc.hmc_step(X, U, dt, 6, gen)
        if float(acc[0]) < 0.5:
            dt = max(dt * 0.85, 0.003)

    r_ref = a_init * 0.5 * (N - 1)
    lim_xy = 1.40 * r_ref
    sigmas, ux, uy, uz, unit_dirs, slice_pts, n_th, n_ph, n_grid = _make_probe_grids(gpu, lim_xy)

    # Compute fixed initial solid-sphere z=0 slice as clean side-by-side reference!
    Y_ref_solid = X[0, 0].to(gpu)
    slice_ref_solid = _batched_point_probe_min_eval(Y_ref_solid, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()

    frames = []
    print(f"[VIEWER v1.0.2] Computing {n_frames} exact U({N})-singlet HMC melting frames (g={g_start:.2f}->{g_end:.2f}, device={gpu})...")
    for fr in range(n_frames):
        # Hold solid for first 12 frames, ramp g across frames 12..56, hold melted 56..71
        if fr < 12:
            g_cur = g_start
        elif fr < 56:
            g_cur = g_start + (g_end - g_start) * ((fr - 12) / 44.0)
        else:
            g_cur = g_end
        hmc.g = float(g_cur)

        Y_live = X[0, 0].to(gpu)
        diag = _geometry_diag(Y_live)
        poly = float(torch.abs(torch.trace(U[0])) / float(N))
        slice_live = _batched_point_probe_min_eval(Y_live, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()
        r_surf_f, lam_surf_f = _batched_3d_shell_surface(Y_live, unit_dirs, sigmas, r_max=lim_xy)
        r_surf = r_surf_f.reshape(n_th, n_ph)
        lam_surf = lam_surf_f.reshape(n_th, n_ph)

        r_north = float(r_surf[0, 0])
        y_probe = torch.tensor([0.0, 0.0, max(r_north, 0.5)], dtype=torch.float64, device=gpu)
        qfim = _exact_qfim_eigenvalues(Y_live, y_probe, sigmas)
        # Normalize QFIM by its largest eigenvalue so rank-2 (0, 1, 1) -> melted 3D/0D breakdown is crystal clear!
        qfim_norm = qfim / max(float(qfim[2]), 1e-12)

        frames.append({
            "x_val": float(g_cur),
            "g": float(g_cur), "nu": float(nu),
            "tr_live": float(diag["tr_Y2"]),
            "r2_evals": np.array(diag["r2_evals"], dtype=np.float64),
            "r2_ctrl": None,
            "alg_err": float(diag["rel_alg_err"]),
            "poly": poly,
            "slice_main": slice_live,
            "slice_ctrl": slice_ref_solid,
            "r_surf": r_surf, "lam_surf": lam_surf,
            "qfim": qfim, "qfim_norm": qfim_norm,
        })
        if fr % 12 == 0 or fr == n_frames - 1:
            print(
                f"  frame {fr+1:2d}/{n_frames} (g={g_cur:4.2f}) | Tr(X^2)={diag['tr_Y2']:7.1f} | "
                f"AlgErr={diag['rel_alg_err']:.3f} | |P|={poly:.2f} | QFIM/g3=[{qfim_norm[0]:.3f}, {qfim_norm[1]:.3f}, {qfim_norm[2]:.3f}]"
            )
        if fr == n_frames - 1:
            break
        acc_sum = 0.0
        for _ in range(7):
            X, U, acc, _ = hmc.hmc_step(X, U, dt, 6, gen)
            acc_sum += float(acc[0])
        ar = acc_sum / 7.0
        if ar < 0.50:
            dt = max(dt * 0.82, 0.002)
        elif ar > 0.86:
            dt = min(dt * 1.12, 0.08)

    guard.cleanup()
    print(f"[VIEWER v1.0.2] Melting trajectory ready in {time.time()-t0:.1f}s!")
    return {
        "mode": "melting", "N": N, "lim_xy": float(lim_xy),
        "ux": ux, "uy": uy, "uz": uz, "frames": frames,
    }


@torch.no_grad()
def _generate_emergence_frames(N: int = 8, n_frames: int = 68, prefer_cuda: bool = True) -> Dict[str, object]:
    """Classic v1.0.1-style spontaneous emergence from 0D GUE noise (with honest v1.0.2 labels, no Phi_TN)."""
    t0 = time.time()
    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    device = guard.device
    core = MatrixNativeQuantumCore(N=N, D=3, guard=guard)
    g, mu, nu_A, nu_B = 0.25, 1.0, 0.80, 3.00
    kappa, g1, g2 = 0.15, 0.015, 1e-4

    gen = torch.Generator(device="cpu").manual_seed(42)
    Y0 = core.random_isotropic_gue(7.0 * (float(N) / 8.0), gen)
    Y_A, Y_B = Y0.clone(), Y0.clone()
    dt_A, pY_A, pG_A = 0.005, None, None
    dt_B, pY_B, pG_B = 0.005, None, None

    alpha_cl = (nu_B + np.sqrt(nu_B * nu_B - 2.0 * mu * mu)) / (2.0 * g)
    lim_xy = 1.40 * alpha_cl * 0.5 * (N - 1)
    sigmas, ux, uy, uz, unit_dirs, slice_pts, n_th, n_ph, n_grid = _make_probe_grids(device, lim_xy)

    def bb_step(Y_cur, dt_cur, pY, pG, nu_val):
        E0, G, _ = core.total_energy_and_grad(Y_cur, g=g, mu=mu, nu=nu_val, kappa=kappa, g1=g1, g2=g2, compute_grad=True)
        gn = float(torch.sqrt(torch.real(torch.sum(G * G.conj()))))
        if gn < 1e-4:
            return Y_cur, dt_cur, pY, pG
        if pY is not None and pG is not None:
            sY, dG = Y_cur - pY, G - pG
            sy, ss = float(torch.real(torch.sum(sY * dG.conj()))), float(torch.real(torch.sum(sY * sY.conj())))
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
    print(f"[VIEWER v1.0.2] Computing {n_frames} emergence frames (N={N}, device={device})...")
    for fr in range(n_frames):
        diag_A, diag_B = _geometry_diag(Y_A), _geometry_diag(Y_B)
        slice_B = _batched_point_probe_min_eval(Y_B, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()
        slice_A = _batched_point_probe_min_eval(Y_A, slice_pts, sigmas).reshape(n_grid, n_grid).cpu().numpy()
        r_surf_f, lam_surf_f = _batched_3d_shell_surface(Y_B, unit_dirs, sigmas, r_max=lim_xy)
        r_surf, lam_surf = r_surf_f.reshape(n_th, n_ph), lam_surf_f.reshape(n_th, n_ph)
        y_probe = torch.tensor([0.0, 0.0, max(float(r_surf[0, 0]), 0.5)], dtype=torch.float64, device=device)
        qfim = _exact_qfim_eigenvalues(Y_B, y_probe, sigmas)
        qfim_norm = qfim / max(float(qfim[2]), 1e-12)
        frames.append({
            "x_val": float(fr * 2),
            "g": g, "nu": nu_B,
            "tr_live": float(diag_B["tr_Y2"]), "tr_ctrl": float(diag_A["tr_Y2"]),
            "r2_evals": np.array(diag_B["r2_evals"], dtype=np.float64),
            "r2_ctrl": float(np.mean(diag_A["r2_evals"])),
            "alg_err": float(diag_B["rel_alg_err"]),
            "poly": None,
            "slice_main": slice_B, "slice_ctrl": slice_A,
            "r_surf": r_surf, "lam_surf": lam_surf,
            "qfim": qfim, "qfim_norm": qfim_norm,
        })
        if fr == n_frames - 1:
            break
        for _ in range(2):
            Y_A, dt_A, pY_A, pG_A = bb_step(Y_A, dt_A, pY_A, pG_A, nu_A)
            Y_B, dt_B, pY_B, pG_B = bb_step(Y_B, dt_B, pY_B, pG_B, nu_B)

    guard.cleanup()
    print(f"[VIEWER v1.0.2] Emergence trajectory ready in {time.time()-t0:.1f}s!")
    return {
        "mode": "emergence", "N": N, "lim_xy": float(lim_xy),
        "ux": ux, "uy": uy, "uz": uz, "frames": frames,
    }


def _launch_classic_5panel_ui(data: Dict[str, object], headless: bool = False, save_gif: str = "", fps: int = 12):
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

    mode = data["mode"]
    frames = data["frames"]
    n_frames = len(frames)
    N = data["N"]
    lim = data["lim_xy"]
    ux, uy, uz = data["ux"], data["uy"], data["uz"]

    xs = np.arange(n_frames, dtype=float) if mode in ("melting", "birth") else np.array([f["x_val"] for f in frames])
    r2_all = np.array([f["r2_evals"] for f in frames])
    qfim_all = np.array([f["qfim_norm"] if mode in ("melting", "birth") else f["qfim"] for f in frames])

    fig = plt.figure(figsize=(15.2, 9.0), dpi=105)
    if mode == "birth":
        title_str = f"Genesis-Rigorous v1.0.2 — Exact U({N})-Singlet HMC Spontaneous Birth of 2D Space (Zero Seed, With QFIM)"
    elif mode == "melting":
        title_str = f"Genesis-Rigorous v1.0.2 — Exact U({N})-Singlet HMC Quantum Melting (No Auto-Rotation, With QFIM)"
    else:
        title_str = f"Genesis-Rigorous v1.0.2 — Spontaneous Fuzzy-Sphere Emergence from 0D GUE Noise (N={N}, U(N)-Covariant)"
    fig.suptitle(title_str, fontsize=12.6, fontweight="bold", color="#fbbf24", y=0.976)
    hud_txt = fig.text(0.5, 0.939, "", ha="center", va="center", fontsize=9.8, color="#93c5fd", family="monospace")

    # Clean 5-panel layout: Top row = 3 equal panels, Bottom row = 2 wide panels (NO Phi_TN panel!)
    gs_top = fig.add_gridspec(1, 3, left=0.04, right=0.97, top=0.872, bottom=0.518, wspace=0.22)
    gs_bot = fig.add_gridspec(1, 2, left=0.06, right=0.96, top=0.392, bottom=0.115 if not headless else 0.075, wspace=0.22)

    ax3d = fig.add_subplot(gs_top[0, 0], projection="3d")
    ax_s1 = fig.add_subplot(gs_top[0, 1])
    ax_s2 = fig.add_subplot(gs_top[0, 2])
    ax_r2 = fig.add_subplot(gs_bot[0, 0])
    ax_qf = fig.add_subplot(gs_bot[0, 1])

    ax3d.set_facecolor("#0b0f19")
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1f2937")

    def log_field(lam_arr):
        return -np.log10(np.clip(lam_arr, 1e-3, None) + 0.08)

    ref_idx = 0 if mode == "melting" else -1
    vmin_s = float(np.percentile(log_field(frames[ref_idx]["slice_main"]), 5))
    vmax_s = float(np.percentile(log_field(frames[ref_idx]["slice_main"]), 99.5))

    im_1 = ax_s1.imshow(log_field(frames[0]["slice_main"]).T, origin="lower", extent=[-lim, lim, -lim, lim], cmap="inferno", vmin=vmin_s, vmax=vmax_s, aspect="equal")
    im_2 = ax_s2.imshow(log_field(frames[0]["slice_ctrl"]).T, origin="lower", extent=[-lim, lim, -lim, lim], cmap="inferno", vmin=vmin_s, vmax=vmax_s, aspect="equal")
    if mode == "birth":
        ax_s1.set_title("2. Live Exact HMC (nu=2.30, z=0 Slice)\n[0D Quantum Fog Crystallizes Into 2D Ring]", fontsize=9.3)
        ax_s2.set_title("3. Parallel HMC Control (nu=0.80, z=0 Slice)\n[Identical Initial Fog -> Stays 0D Vacuum]", fontsize=9.3)
    elif mode == "melting":
        ax_s1.set_title("2. Live Exact HMC Slice (z=0): -log10(|lambda_min|)\n[Sharp Golden Ring Shrinks & Melts Into Fog]", fontsize=9.3)
        ax_s2.set_title("3. Reference Solid Sphere (g=0.24, z=0 Slice)\n[Fixed Baseline For Direct Visual Comparison]", fontsize=9.3)
    else:
        ax_s1.set_title("2. Universe B (nu=3.0): Exact z=0 Dirac Field\n[0D Noise Crystallizes Into Sharp Ring]", fontsize=9.3)
        ax_s2.set_title("3. Trivial Vacuum Control (nu=0.8): z=0 Slice\n[Identical Random GUE Seed -> 0D Collapse]", fontsize=9.3)
    ax_s1.set_xlabel("x"); ax_s1.set_ylabel("y")
    ax_s2.set_xlabel("x"); ax_s2.set_ylabel("y")

    # Panel 4 (Bottom-Left, wide): R^2 Eigenvalues
    r2_lines = [ax_r2.plot([], [], lw=1.8, alpha=0.88, color=plt.cm.cool(i / max(N - 1, 1)))[0] for i in range(N)]
    r2_ctrl_line = None
    poly_line = None
    if mode in ("emergence", "birth"):
        r2_ctrl_line, = ax_r2.plot([], [], "r--", lw=1.9, label="Sub-Critical Control Mean R^2 (0D)")
    if mode in ("melting", "birth"):
        ax_r2_p = ax_r2.twinx()
        poly_all = np.array([f["poly"] for f in frames])
        poly_line, = ax_r2_p.plot([], [], color="#34d399", ls="--", lw=2.0, label="Polyakov Loop |Tr U|/N (Right)")
        ax_r2_p.set_ylim(0.0, 1.12)
        ax_r2_p.set_ylabel("Polyakov Loop |Tr U|/N", color="#34d399")
    ax_r2.set_xlim(xs[0], xs[-1])
    ax_r2.set_ylim(-0.04 * np.max(r2_all), 1.12 * np.max(r2_all))
    if mode == "birth":
        ax_r2.set_title("4. R^2 Eigenvalues & Polyakov Loop During Quantum Birth\n[0D Liquid Unfreezes & Snaps Into Single 2D Sphere Shell]", fontsize=9.5)
        ax_r2.set_xlabel("HMC Cooling Frame (g: 1.35 -> 0.25)")
    elif mode == "melting":
        ax_r2.set_title("4. R^2 = sum(X_I^2) Eigenvalues & Polyakov Loop\n[Tightly Bundled Sphere Shells Collapse & Scatter Under Melting]", fontsize=9.5)
        ax_r2.set_xlabel("HMC Melting Frame (g: 0.24 -> 1.30)")
    else:
        ax_r2.set_title("4. R^2 = sum(Y_I^2) Eigenvalue Degeneracy\n[N Scattered Noise Lines Lock Into Smooth Sphere]", fontsize=9.5)
        ax_r2.set_xlabel("Relaxation Step (BB Flow)")
    ax_r2.set_ylabel("Eigenvalues of R^2")
    if mode == "emergence":
        ax_r2.legend(loc="upper left", fontsize=8.0)
    elif mode == "birth":
        h1, l1 = ax_r2.get_legend_handles_labels()
        h2, l2 = ax_r2_p.get_legend_handles_labels()
        ax_r2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.0)
    else:
        ax_r2_p.legend(loc="upper right", fontsize=8.0)
    ax_r2.grid(True, alpha=0.35)

    # Panel 5 (Bottom-Right, wide): Quantum Information Geometry (QFIM = 4 * g_PV)
    q_cols = ["#f87171", "#34d399", "#60a5fa"]
    q_lbls = ["g_1 (Normal: 0 on 2D Sphere)", "g_2 (Tangent 1)", "g_3 (Tangent 2)"]
    qf_lines = [ax_qf.plot([], [], lw=2.2, color=q_cols[k], label=q_lbls[k])[0] for k in range(3)]
    ax_qf.set_xlim(xs[0], xs[-1])
    if mode == "birth":
        ax_qf.set_ylim(-0.05, 1.15)
        ax_qf.set_title("5. Quantum Information Geometry (Normalized QFIM = g_a / g_3)\n[Quantum Fog Chaos -> Exact 2D Rank-2 Manifold (0, 1, 1)]", fontsize=9.5)
        ax_qf.set_xlabel("HMC Cooling Frame (g: 1.35 -> 0.25)")
        ax_qf.set_ylabel("Normalized QFIM Eigenvalues")
    elif mode == "melting":
        ax_qf.set_ylim(-0.05, 1.15)
        ax_qf.set_title("5. Quantum Information Geometry (Normalized QFIM = g_a / g_3)\n[Clean 2D Rank-2 (0, 1, 1) Breaks Down When Space Melts]", fontsize=9.5)
        ax_qf.set_xlabel("HMC Melting Frame (g: 0.24 -> 1.30)")
        ax_qf.set_ylabel("Normalized QFIM Eigenvalues")
    else:
        q_max_vis = float(np.percentile(qfim_all[:, 2], 96)) * 1.25 + 1e-4
        ax_qf.set_ylim(-0.05 * q_max_vis, q_max_vis)
        ax_qf.set_title("5. Quantum Information Geometry (QFIM = 4 * g_PV)\n[3D Chaos -> Exact Rank-2 2D Manifold (0, g_t, g_t)]", fontsize=9.5)
        ax_qf.set_xlabel("Relaxation Step (BB Flow)")
        ax_qf.set_ylabel("QFIM Eigenvalues")
    ax_qf.legend(loc="center right" if mode in ("melting", "birth") else "upper right", fontsize=8.0)
    ax_qf.grid(True, alpha=0.35)

    state = {"idx": 0, "playing": True}

    def render_frame(idx: int):
        f = frames[idx]
        # Panel 1: STATIC 3D Point-Probe Dirac Shell (NO auto-rotation! User can drag with mouse.)
        elev_cur = ax3d.elev if ax3d.elev is not None else 24.0
        azim_cur = ax3d.azim if ax3d.azim is not None else -55.0
        ax3d.cla()
        ax3d.set_facecolor("#0b0f19")
        r_s, lam_s = f["r_surf"], f["lam_surf"]
        X3, Y3, Z3 = r_s * ux, r_s * uy, r_s * uz

        # Per-facet outward normals & directional lighting so every bump, ripple, and curvature stands out!
        P = np.stack([X3, Y3, Z3], axis=-1)
        d_th = 0.5 * ((P[1:, :-1] - P[:-1, :-1]) + (P[1:, 1:] - P[:-1, 1:]))
        d_ph = 0.5 * ((P[:-1, 1:] - P[:-1, :-1]) + (P[1:, 1:] - P[1:, :-1]))
        nrm = np.cross(d_th, d_ph)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)

        er, ar = np.deg2rad(elev_cur), np.deg2rad(azim_cur)
        v_cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
        l_key = np.array([0.15, -0.78, 0.61]); l_key /= np.linalg.norm(l_key)
        l_fill = np.array([-0.68, 0.58, -0.45]); l_fill /= np.linalg.norm(l_fill)
        h_vec = l_key + v_cam; h_vec /= max(np.linalg.norm(h_vec), 1e-12)

        ndotl = np.clip(np.sum(nrm * l_key, axis=-1), 0.0, 1.0)
        fill = np.clip(np.sum(nrm * l_fill, axis=-1), 0.0, 1.0)
        spec = np.clip(np.sum(nrm * h_vec, axis=-1), 0.0, 1.0) ** 14

        sharp = 1.0 / (1.0 + lam_s / max(0.15 * lim, 1.0))
        sharp_f = 0.25 * (sharp[:-1, :-1] + sharp[1:, :-1] + sharp[:-1, 1:] + sharp[1:, 1:])
        r_f = 0.25 * (r_s[:-1, :-1] + r_s[1:, :-1] + r_s[:-1, 1:] + r_s[1:, 1:])
        r_mean = max(float(np.mean(r_f)), 1e-6)
        r_dev = np.clip((r_f - r_mean) / max(0.20 * r_mean, 0.25), -1.0, 1.0)

        # Warm golden-amber base (matching inferno slices) with rich facet-to-facet tonal shading
        c_val = np.clip(0.86 * sharp_f + 0.14 * r_dev, 0.28, 0.98)
        base_rgb = plt.cm.inferno(c_val)[..., :3]
        # Boost warm golden tone slightly so lit facets glow rich yellow-gold while shadow facets turn deep bronze
        base_rgb = np.clip(base_rgb * np.array([1.05, 1.02, 0.85]), 0.0, 1.0)
        light = 0.26 + 0.78 * (ndotl ** 0.90) + 0.15 * fill
        shaded_rgb = np.clip(
            base_rgb * light[..., None] + 0.30 * spec[..., None] * np.array([1.0, 0.97, 0.82]),
            0.0, 1.0,
        )
        face_cols = np.concatenate([shaded_rgb, np.full(shaded_rgb.shape[:2] + (1,), 0.96)], axis=-1)

        ax3d.plot_surface(
            X3, Y3, Z3, facecolors=face_cols, rstride=1, cstride=1,
            shade=False, antialiased=True, linewidth=0.32, edgecolor="#1c110666",
        )
        lim_3d = 0.84 * lim
        ax3d.set_xlim(-lim_3d, lim_3d); ax3d.set_ylim(-lim_3d, lim_3d); ax3d.set_zlim(-lim_3d, lim_3d)
        try:
            ax3d.set_box_aspect([1, 1, 1], zoom=1.22)
        except TypeError:
            ax3d.set_box_aspect([1, 1, 1])
        ax3d.view_init(elev=elev_cur, azim=azim_cur)
        ax3d.set_title("1. 3D Point-Probe Dirac Shell (Static Camera)\n[Zero Mesh Cheat | Shaded Facets = Curvature & Ripples]", fontsize=9.3)
        ax3d.set_xlabel("X", labelpad=-3); ax3d.set_ylabel("Y", labelpad=-3); ax3d.set_zlabel("Z", labelpad=-3)
        ax3d.tick_params(labelsize=7, pad=-2)

        # Panels 2 & 3: 2D z=0 inferno slices
        im_1.set_data(log_field(f["slice_main"]).T)
        im_2.set_data(log_field(f["slice_ctrl"]).T)

        # Panels 4 & 5: Time-series
        sl = slice(0, idx + 1)
        x_sl = xs[sl]
        for i in range(N):
            r2_lines[i].set_data(x_sl, r2_all[sl, i])
        if r2_ctrl_line is not None:
            r2_ctrl_line.set_data(x_sl, [fr_i["r2_ctrl"] for fr_i in frames[: idx + 1]])
        if poly_line is not None:
            poly_line.set_data(x_sl, poly_all[sl])
        for k in range(3):
            qf_lines[k].set_data(x_sl, qfim_all[sl, k])

        qn = f["qfim_norm"]
        eff_rank = int(np.sum(qn > 0.15))
        if mode in ("melting", "birth"):
            hud_txt.set_text(
                f"Frame {idx+1:2d}/{n_frames}  |  nu={f['nu']:.2f}, g={f['g']:.2f}  |  "
                f"Tr(X^2)={f['tr_live']:,.1f}  |  AlgErr={f['alg_err']:.3f}  |  "
                f"Polyakov |P|={f['poly']:.2f}  |  QFIM/g3=[{qn[0]:.2f}, {qn[1]:.2f}, 1.00]"
            )
        else:
            hud_txt.set_text(
                f"Frame {idx+1:2d}/{n_frames}  |  Step={int(f['x_val']):3d}  |  "
                f"Tr(Y^2): Ctrl={f['tr_ctrl']:.1f} vs Univ B={f['tr_live']:,.1f}  |  "
                f"AlgErr={f['alg_err']:.2e}  |  QFIM Rank={eff_rank}D"
            )
        return []

    ax3d.view_init(elev=24.0, azim=-55.0)
    render_frame(0)

    if save_gif:
        os.makedirs(os.path.dirname(save_gif) or ".", exist_ok=True)
        print(f"[VIEWER v1.0.2] Exporting classic 5-panel GIF -> {save_gif} ...")
        ani_save = FuncAnimation(fig, lambda i: render_frame(i), frames=n_frames, interval=int(1000 / fps), blit=False)
        ani_save.save(save_gif, writer=PillowWriter(fps=fps))
        print(f"[VIEWER v1.0.2] Saved GIF: {save_gif}")

    if headless:
        plt.close(fig)
        return

    ax_sl = fig.add_axes([0.14, 0.032, 0.54, 0.028], facecolor="#1f2937")
    slider = Slider(ax_sl, "Frame", 0, n_frames - 1, valinit=0, valstep=1, color="#38bdf8")
    ax_btn_pp = fig.add_axes([0.72, 0.028, 0.10, 0.04])
    btn_pp = Button(ax_btn_pp, "Pause", color="#1e3a8a", hovercolor="#2563eb")
    ax_btn_rp = fig.add_axes([0.84, 0.028, 0.10, 0.04])
    btn_rp = Button(ax_btn_rp, "Replay", color="#065f46", hovercolor="#059669")

    updating = {"flag": False}

    def on_slider(val):
        if updating["flag"]:
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
        updating["flag"] = True
        slider.set_val(0)
        updating["flag"] = False
        render_frame(0)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)
    btn_pp.on_clicked(on_pp)
    btn_rp.on_clicked(on_rp)

    def step_anim(_f):
        if not state["playing"]:
            return []
        state["idx"] = (state["idx"] + 1) % n_frames
        updating["flag"] = True
        slider.set_val(state["idx"])
        updating["flag"] = False
        return render_frame(state["idx"])

    _ani = FuncAnimation(fig, step_anim, interval=int(1000 / fps), blit=False, cache_frame_data=False)
    plt.show()


def run_v102_classic_viewer(
    mode: str = "birth",
    N: int = 0,
    frames: int = 72,
    save_gif: str = "",
    headless: bool = False,
    fps: int = 12,
    prefer_cuda: bool = True,
) -> None:
    if mode == "birth":
        data = _generate_birth_frames(N=(N if N > 0 else 4), n_frames=frames, prefer_cuda=prefer_cuda)
    elif mode == "melting":
        data = _generate_melting_frames(N=(N if N > 0 else 6), n_frames=frames, prefer_cuda=prefer_cuda)
    elif mode == "emergence":
        data = _generate_emergence_frames(N=(N if N > 0 else 8), n_frames=frames, prefer_cuda=prefer_cuda)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'birth', 'melting', or 'emergence'.")
    _launch_classic_5panel_ui(data, headless=headless or bool(save_gif), save_gif=save_gif, fps=fps)
