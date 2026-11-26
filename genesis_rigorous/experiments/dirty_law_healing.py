"""
Level-2 Experiment L2-B: Dirty-Law Healing & Large-N Robustness of Emergent Spacetime
=====================================================================================
Tests H. B. Nielsen's "Random Dynamics" / Large-N RG Robustness hypothesis directly:
If the microscopic physical law is deliberately UGLY and BROKEN (violating SO(3) rotation
symmetry, breaking gauge-commutator structure, or adding parity-breaking symmetric cubics),
what happens as matrix size N grows (N = 6, 8, 12, 16, 20)?

Does emergent 2D spacetime:
  (a) Survive and become *sharper* in Quantum Information Geometry (g_1/g_3 -> 0, blur -> 0)?
  (b) Self-heal its SO(3) shape, stay a smooth warped 2D ellipsoid, or shatter into multi-branes?
  (c) Preserve the Ryu-Takayanagi / link-entanglement Area Law r(S_TN, sqrt(M(N-M))) even
      under severely broken microscopic laws?

Five Controlled Microscopic Laws Tested Across N = 6, 8, 12, 16, 20:
  L0 (Clean YM+Myers Baseline)   : Exact SO(3)-symmetric benchmark
  L1 (3.5x Anisotropic Mass)     : M = diag(0.6, 1.3, 2.1) — strongly breaks SO(3) in V_2
  L2 (1.6x Anisotropic YM)       : g = (0.20, 0.25, 0.32) — strongly breaks SO(3) in gauge kinetic term
  L3 (Parity-Broken Sym Cubic)   : Clean YM+Myers + random symmetric cubic S_IJK
  L4 (Non-Gauge Quartic, 't Hooft): Clean YM+Myers + (q_0 / N^2) * sum_I Tr(X_I^4) vs unscaled q_0!
"""

import datetime
import json
import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..safety import HardwareSafetyGuard, SafetyConfig, safe_eigh, safe_eigvalsh
from .law_ecosystem import (
    GeneralRandomMatrixLaw,
    classify_emergent_geometry,
    _random_sym_cubic_3x3x3,
)


@torch.no_grad()
def compute_area_law_corr(Y: torch.Tensor, g1: float = 0.015, g2: float = 1e-4) -> float:
    """
    Computes exact unsymmetrized link-fermion entanglement S_TN(M) across ranks M=1..N-1
    along the principal axis of Y (top eigenvector of Cov_IJ = Tr(Y_I Y_J)), and returns
    its Pearson correlation with the 2D cap perimeter sqrt(M(N-M)).
    """
    N = Y.shape[1]
    device = Y.device
    # Principal axis (100% SO(3) & U(N) covariant!)
    cov = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            cov[i, j] = float(torch.real(torch.sum(Y[i] * Y[j].conj())))
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, 2]  # Longest principal axis
    Y_ax = axis[0] * Y[0] + axis[1] * Y[1] + axis[2] * Y[2]
    _, U_ax = safe_eigh(0.5 * (Y_ax + Y_ax.conj().T))

    eye_N = torch.eye(N, dtype=torch.complex128, device=device)
    Delta = torch.zeros((N * N, N * N), dtype=torch.complex128, device=device)
    for I in range(3):
        YI = Y[I].contiguous()
        Y2 = (YI @ YI).contiguous()
        Delta += torch.kron(Y2, eye_N) + torch.kron(eye_N, Y2.T.contiguous()) - 2.0 * torch.kron(YI, YI.T.contiguous())
    Delta = 0.5 * (Delta + Delta.conj().T)
    lam, U_eig = safe_eigh(Delta)
    lam = torch.clamp(lam, min=0.0)
    f_v = g1 * lam + g2 * (lam * lam)
    eps_v = torch.sqrt(f_v * f_v + 1.0)
    n_k = (eps_v - f_v) / (2.0 * eps_v)
    m_k = (eps_v + f_v) / (2.0 * eps_v)
    a_k = -1j / (2.0 * eps_v)
    U_mats = U_eig.T.reshape(N * N, N, N)

    s_vals, perim_vals = [], []
    for M in range(1, N):
        Q_in = U_ax[:, N - M:]
        B = torch.einsum("ia,kab,bj->kij", Q_in.conj().T, U_mats, Q_in).reshape(N * N, M * M)
        G11 = B.conj().T @ (n_k.to(torch.complex128)[:, None] * B)
        G22 = B.conj().T @ (m_k.to(torch.complex128)[:, None] * B)
        G21 = B.conj().T @ (a_k.to(torch.complex128)[:, None] * B)
        G_blk = torch.cat([torch.cat([G11, G21.conj().T], dim=1), torch.cat([G21, G22], dim=1)], dim=0)
        G_blk = 0.5 * (G_blk + G_blk.conj().T)
        ev = torch.clamp(safe_eigvalsh(G_blk), min=1e-13, max=1.0 - 1e-13)
        s_m = float(-torch.sum(ev * torch.log(ev) + (1.0 - ev) * torch.log(1.0 - ev)))
        s_vals.append(s_m)
        perim_vals.append(np.sqrt(M * (N - M)))
    if np.std(s_vals) < 1e-12:
        return 0.0
    return float(np.corrcoef(s_vals, perim_vals)[0, 1])


@torch.no_grad()
def relax_to_global_basin(law: GeneralRandomMatrixLaw, N: int, device: torch.device) -> torch.Tensor:
    """Relax from multi-scale isotropic GUE clouds (picking lowest-energy equilibrium)."""
    best_Y, best_E = None, float("inf")
    scale_list = [6.5 * (N / 8.0), 9.0 * (N / 8.0)]
    for idx_s, sc in enumerate(scale_list):
        gen = torch.Generator(device="cpu").manual_seed(42 + 100 * idx_s)
        Z = (torch.randn((3, N, N), generator=gen, dtype=torch.float64) + 1j * torch.randn((3, N, N), generator=gen, dtype=torch.float64)).to(device)
        Y = law.project_herm_traceless(Z)
        fn = float(torch.sqrt(torch.real(torch.sum(Y * Y.conj()))))
        Y = Y * (sc * np.sqrt(3.0 * N) / max(fn, 1e-12))
        dt, pY, pG = 0.006, None, None
        for _ in range(150):
            E0, G = law.energy_and_grad(Y, compute_grad=True)
            gn = float(torch.sqrt(torch.real(torch.sum(G * G.conj()))))
            if gn < 1e-3:
                break
            if pY is not None and pG is not None:
                sY, dG = Y - pY, G - pG
                sy = float(torch.real(torch.sum(sY * dG.conj())))
                ss = float(torch.real(torch.sum(sY * sY.conj())))
                if sy > 1e-12 and ss > 1e-16:
                    dt = float(np.clip(ss / sy, 1e-5, 0.12))
            yn = float(torch.sqrt(torch.real(torch.sum(Y * Y.conj()))))
            ssz = min(dt, max(2.0, 0.22 * yn) / max(gn, 1.0))
            acc = False
            for _ in range(8):
                Yt = law.project_herm_traceless(Y - ssz * G)
                Et, _ = law.energy_and_grad(Yt, compute_grad=False)
                if Et <= E0 - 1e-4 * ssz * gn * gn:
                    pY, pG, Y = Y.clone(), G.clone(), Yt
                    dt = min(ssz * 1.35, 0.12); acc = True; break
                ssz *= 0.4
            if not acc:
                dt *= 0.3; pY, pG = None, None
        Ef, _ = law.energy_and_grad(Y, compute_grad=False)
        if Ef < best_E:
            best_E, best_Y = Ef, Y.clone()
    return best_Y


@torch.no_grad()
def run_dirty_law_healing_experiment(Ns: List[int] = None, out_dir: str = "", prefer_cuda: bool = True) -> Dict[str, object]:
    if Ns is None:
        Ns = [6, 8, 12, 16]
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join("results", f"l2b_dirty_law_healing_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    device = guard.device

    k_idx = np.arange(64, dtype=np.float64) + 0.5
    phi_fib = np.arccos(1.0 - 2.0 * k_idx / 64.0)
    th_fib = np.pi * (1.0 + 5.0 ** 0.5) * k_idx
    unit_dirs = torch.tensor(
        np.stack([np.sin(phi_fib) * np.cos(th_fib), np.sin(phi_fib) * np.sin(th_fib), np.cos(phi_fib)], axis=-1),
        dtype=torch.float64, device=device,
    )
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    sigmas = torch.stack([s1, s2, s3], dim=0)

    rng_fix = np.random.default_rng(2026)
    S_dirty = _random_sym_cubic_3x3x3(rng_fix, 0.04)

    law_defs = [
        ("L0_Clean_SO3", "L0: Clean SO(3) YM+Myers"),
        ("L1_Aniso_Mass", "L1: 3.5x Anisotropic Mass M=diag(0.6,1.3,2.1)"),
        ("L2_Aniso_YM", "L2: 1.6x Anisotropic YM g=(0.20,0.25,0.32)"),
        ("L3_Broken_Cubic", "L3: Parity-Broken Random Sym Cubic"),
        ("L4_NonGauge_Quartic_Unscaled", "L4: Unscaled Non-Gauge Quartic q*Tr(X^4)"),
        ("L5_NonGauge_Quartic_tHooft", "L5: 't Hooft-Scaled Quartic (q/N^2)*Tr(X^4)"),
    ]

    results = {k: [] for k, _ in law_defs}

    for N in Ns:
        print(f"\n[EXP L2-B] Testing Dirty-Law Survival & Healing at N = {N}...")
        for lk, ltitle in law_defs:
            if lk == "L0_Clean_SO3":
                law = GeneralRandomMatrixLaw(np.eye(3), np.zeros((3, 3, 3)), 0.75, np.array([0.25, 0.25, 0.25]), np.zeros(3), np.zeros(3), device)
            elif lk == "L1_Aniso_Mass":
                law = GeneralRandomMatrixLaw(np.diag([0.6, 1.3, 2.1]), np.zeros((3, 3, 3)), 0.75, np.array([0.25, 0.25, 0.25]), np.zeros(3), np.zeros(3), device)
            elif lk == "L2_Aniso_YM":
                law = GeneralRandomMatrixLaw(np.eye(3), np.zeros((3, 3, 3)), 0.75, np.array([0.20, 0.25, 0.32]), np.zeros(3), np.zeros(3), device)
            elif lk == "L3_Broken_Cubic":
                law = GeneralRandomMatrixLaw(np.eye(3), S_dirty, 0.75, np.array([0.25, 0.25, 0.25]), np.full(3, 5e-4), np.zeros(3), device)
            elif lk == "L4_NonGauge_Quartic_Unscaled":
                law = GeneralRandomMatrixLaw(np.eye(3), np.zeros((3, 3, 3)), 0.75, np.array([0.25, 0.25, 0.25]), np.full(3, 0.008), np.zeros(3), device)
            else:  # L5_NonGauge_Quartic_tHooft
                q_sc = 0.008 * ((6.0 / float(N)) ** 2)
                law = GeneralRandomMatrixLaw(np.eye(3), np.zeros((3, 3, 3)), 0.75, np.array([0.25, 0.25, 0.25]), np.full(3, q_sc), np.zeros(3), device)

            Y_eq = relax_to_global_basin(law, N=N, device=device)
            diag = classify_emergent_geometry(Y_eq, unit_dirs, sigmas)
            r_area = compute_area_law_corr(Y_eq) if diag["tr_Y2"] >= 1.0 else 0.0
            rec = {
                "N": N, "phase": diag["phase"], "sharpness": diag["sharpness"],
                "qfim_g1_g3": diag["g_ratios"][0], "qfim_g2_g3": diag["g_ratios"][1],
                "isotropy": diag["isotropy"], "noncomm": diag["noncomm"],
                "area_law_r": r_area, "tr_Y2": diag["tr_Y2"],
            }
            results[lk].append(rec)
            print(
                f"  [{lk:28s}] {diag['phase']:20s} | blur={diag['sharpness']:.4f} | "
                f"QFIM g1/g3={diag['g_ratios'][0]:.4f} | iso={diag['isotropy']:.3f} | AreaLaw r={r_area:.4f}"
            )

    # --- Plot 4-Panel Scaling Dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.8), dpi=140)
    fig.suptitle("Level-2 Experiment L2-B: Dirty-Law Healing & Large-N RG Robustness (N = 6, 8, 12, 16)", fontsize=13, fontweight="bold", y=0.98)
    cols = ["#111827", "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"]
    mks = ["o", "s", "^", "D", "X", "v"]

    for (lk, ltitle), c, mk in zip(law_defs, cols, mks):
        recs = results[lk]
        ns = [r["N"] for r in recs]
        axes[0, 0].plot(ns, [r["sharpness"] for r in recs], marker=mk, color=c, lw=2.0, ms=6.5, label=ltitle)
        axes[0, 1].plot(ns, [r["qfim_g1_g3"] for r in recs], marker=mk, color=c, lw=2.0, ms=6.5, label=ltitle)
        axes[1, 0].plot(ns, [r["isotropy"] for r in recs], marker=mk, color=c, lw=2.0, ms=6.5, label=ltitle)
        axes[1, 1].plot(ns, [r["area_law_r"] for r in recs], marker=mk, color=c, lw=2.0, ms=6.5, label=ltitle)

    axes[0, 0].axhline(0.08, color="gray", ls="--", lw=1.1)
    axes[0, 0].set_title("1. Manifold Sharpness vs N (Lower = Sharper Spacetime)\n[Does Broken-Law Blur Shrink as N Grows?]", fontsize=9.8)
    axes[0, 0].set_xlabel("Matrix Size N"); axes[0, 0].set_ylabel("Median |lambda_min| / r_rms"); axes[0, 0].legend(fontsize=7.3); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axhline(0.08, color="gray", ls="--", lw=1.1)
    axes[0, 1].set_title("2. Quantum Fisher Metric Normal Ratio g_1 / g_3 -> 0\n[Exact 2D Manifold Dimension Under Broken Laws]", fontsize=9.8)
    axes[0, 1].set_xlabel("Matrix Size N"); axes[0, 1].set_ylabel("QFIM Normal / Tangent Ratio (g_1 / g_3)"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("3. Shell Shape Isotropy (r_10% / r_90%)\n[Warped Ellipsoid vs Self-Healing vs Shattering]", fontsize=9.8)
    axes[1, 0].set_xlabel("Matrix Size N"); axes[1, 0].set_ylabel("Shell Isotropy Ratio"); axes[1, 0].set_ylim(0, 1.08); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("4. Ryu-Takayanagi Link Area-Law Survival r(S_TN, Perim)\n[Does Area Law Survive Broken Microscopic Symmetry?]", fontsize=9.8)
    axes[1, 1].set_xlabel("Matrix Size N"); axes[1, 1].set_ylabel("Pearson r(S_TN, sqrt(M(N-M)))"); axes[1, 1].set_ylim(0.5, 1.02); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = os.path.join(out_dir, "exp_l2b_dirty_law_healing.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)

    report = {
        "Ns": Ns, "timestamp": ts, "out_dir": out_dir, "png_path": png_path,
        "results": results, "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(out_dir, "exp_l2b_dirty_law_healing.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    guard.cleanup()
    print(f"\n[EXP L2-B COMPLETE] Elapsed: {time.time()-t0:.1f}s | Dashboard: {png_path}")
    return report


if __name__ == "__main__":
    run_dirty_law_healing_experiment()
