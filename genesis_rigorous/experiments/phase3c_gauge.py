"""
Phase 3C Experiment: Emergent Non-Abelian U(2) Gauge Bosons & Exact Matrix Higgs Mechanism
U(2) -> U(1) x U(1) (M_W^2 = g^2 v^2) from Coincident Fuzzy Branes (k=2, N=2*n_sphere, D=4).
"""

import datetime
import json
import os
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..exact_hfb import build_hermitian_basis, build_su2_irrep
from ..safety import HardwareSafetyGuard, SafetyConfig


@torch.no_grad()
def build_exact_4d_hessian_matrix_free(
    Y: torch.Tensor, T_basis: torch.Tensor, g: float = 0.25, mu: float = 1.0, nu: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    D, N, _ = Y.shape
    N2, nB = N * N, D * N * N
    device = Y.device

    F_mats = []
    for I in range(D):
        comm = Y[I].unsqueeze(0) @ T_basis - T_basis @ Y[I].unsqueeze(0)
        F_I = torch.real(-1j * torch.einsum("aij,bji->ab", comm, T_basis))
        F_mats.append(0.5 * (F_I - F_I.T))

    H2 = torch.zeros((D, N2, D, N2), dtype=torch.float64, device=device)
    eye_N2 = torch.eye(N2, dtype=torch.float64, device=device)
    g2 = g * g

    for I in range(D):
        H2[I, :, I, :] += (mu * mu) * eye_N2
        for J in range(D):
            if I == J: continue
            H2[I, :, I, :] += g2 * (F_mats[J] @ F_mats[J].T)
            C_IJ_mat = -1j * (Y[I] @ Y[J] - Y[J] @ Y[I])
            comm_C = C_IJ_mat.unsqueeze(0) @ T_basis - T_basis @ C_IJ_mat.unsqueeze(0)
            c_f = torch.real(-1j * torch.einsum("aij,bji->ab", comm_C, T_basis))
            H2[I, :, J, :] += g2 * (-(F_mats[J] @ F_mats[I].T) + c_f)

    eps3 = [(0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0), (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0)]
    for I, J, K, sgn in eps3:
        H2[I, :, J, :] += (-2.0 * nu * g * sgn) * F_mats[K]

    H2_flat = 0.5 * (H2.reshape(nB, nB) + H2.reshape(nB, nB).T)
    G_orb = torch.stack(F_mats, dim=0).permute(1, 0, 2).reshape(N2, nB).T
    return H2_flat, G_orb


@torch.no_grad()
def run_phase3c_gauge_experiment(
    n_sphere: int = 6, g: float = 0.25, mu: float = 1.0, nu: float = 3.0,
    vevs: List[float] = None, out_dir: str = "", prefer_cuda: bool = True
) -> Dict[str, object]:
    if vevs is None:
        vevs = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    N, D = 2 * n_sphere, 4
    out_dir = out_dir or os.path.join("results", f"unified_phase3c_gauge_N{N}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    device = guard.device
    N2, nB = N * N, D * N * N
    L_n = build_su2_irrep(n_sphere, device)
    T_N = build_hermitian_basis(N, device)
    alpha_star = (nu + np.sqrt(nu * nu - 2.0 * mu * mu)) / (2.0 * g)

    eye_n = torch.eye(n_sphere, dtype=torch.complex128, device=device)
    eye_2 = torch.eye(2, dtype=torch.complex128, device=device)
    sig_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    tau_mats = [
        torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128, device=device) / np.sqrt(2.0),
        torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device) / np.sqrt(2.0),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device) / np.sqrt(2.0),
        torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device) / np.sqrt(2.0),
    ]

    def classify_mode(v_vec: torch.Tensor) -> Tuple[float, float, float]:
        V_m = torch.einsum("da,aij->dij", v_vec.reshape(D, N2).to(torch.complex128), T_N)
        n_tot = float(torch.real(torch.sum(V_m * V_m.conj())))
        n_sp = float(torch.real(torch.sum(V_m[:3] * V_m[:3].conj())))
        w_off = float(torch.real(torch.sum(V_m[:, :n_sphere, n_sphere:].abs() ** 2) + torch.sum(V_m[:, n_sphere:, :n_sphere].abs() ** 2))) / max(n_tot, 1e-15)
        return 1.0 - w_off, w_off, n_sp / max(n_tot, 1e-15)

    sweep, om2_W0, om2_ph0, mult_spread = [], None, None, 0.0
    for v in vevs:
        Y = torch.zeros((D, N, N), dtype=torch.complex128, device=device)
        for i in range(3): Y[i] = torch.kron(eye_2, alpha_star * L_n[i])
        Y[3] = 0.5 * v * torch.kron(sig_z, eye_n)

        H2, G_orb = build_exact_4d_hessian_matrix_free(Y, T_N, g=g, mu=mu, nu=nu)
        U_g, s_g, _ = torch.linalg.svd(G_orb, full_matrices=False)
        r_g = int(torch.sum(s_g > 1e-7 * torch.max(s_g)))
        P_phys = torch.eye(nB, dtype=torch.float64, device=device) - U_g[:, :r_g] @ U_g[:, :r_g].T
        ep, up = torch.linalg.eigh(0.5 * (P_phys + P_phys.T))
        Q_phys = up[:, ep > 0.5]
        H_p = Q_phys.T @ H2 @ Q_phys
        ew_p, ev_p = torch.linalg.eigh(0.5 * (H_p + H_p.T))
        V_p = Q_phys @ ev_p

        diag_sp, off_sp = [], []
        for idx in range(ew_p.shape[0]):
            w_d, w_o, w_sp = classify_mode(V_p[:, idx])
            val = float(ew_p[idx])
            if w_sp > 0.80:
                if w_d > 0.80: diag_sp.append(val)
                elif w_o > 0.80: off_sp.append(val)
                else: diag_sp.append(val); off_sp.append(val)
        diag_sp.sort(); off_sp.sort()
        if v == 0.0:
            om2_ph0, om2_W0 = diag_sp[0], off_sp[0]
            sp_all = sorted([float(ew_p[i]) for i in range(ew_p.shape[0]) if classify_mode(V_p[:, i])[2] > 0.80])
            mult_spread = float(max(sp_all[:12]) - min(sp_all[:12]))
        dW, dPh, ex = off_sp[0] - om2_W0, diag_sp[0] - om2_ph0, (g * v) ** 2
        kk_diag = [x for x in diag_sp if x > om2_ph0 + 1.0]
        kk_off = [x for x in off_sp if x > off_sp[0] + 1.0]
        print(f"  [3C v={v:4.1f}] Phys={ew_p.shape[0]} | Photon d(om^2)={dPh:.2e} | W+- d(om^2)={dW:.5f} vs g^2 v^2={ex:.5f} (err={abs(dW-ex):.2e})")
        sweep.append({
            "vev": v, "n_phys": int(ew_p.shape[0]), "n_gauge": int(r_g),
            "photon_om2": diag_sp[0], "photon_shift": dPh,
            "W_om2": off_sp[0], "W_shift": dW, "exact_g2_v2": ex, "err": abs(dW - ex),
            "kk_diag_om2": kk_diag[0] if kk_diag else float("nan"),
            "kk_off_om2": kk_off[0] if kk_off else float("nan"),
        })

    # Non-Abelian SU(2) vs Abelian U(1) quartic self-interaction
    Y0 = torch.stack([torch.kron(eye_2, alpha_star * L_n[i]) for i in range(3)], dim=0)
    def ym_e(Ym):
        return sum(0.5 * g * g * float(torch.real(torch.sum((Ym[I] @ Ym[J] - Ym[J] @ Ym[I]).abs() ** 2))) for I in range(3) for J in range(I + 1, 3))
    a_sp = L_n[2] / torch.sqrt(torch.real(torch.sum(L_n[2] * L_n[2].conj())))
    A1s, A2s = torch.zeros_like(Y0), torch.zeros_like(Y0)
    A1s[0], A2s[1] = torch.kron(tau_mats[2], a_sp), torch.kron(tau_mats[3], a_sp)
    A1u, A2u = torch.zeros_like(Y0), torch.zeros_like(Y0)
    A1u[0], A2u[1] = torch.kron(tau_mats[0], a_sp), torch.kron(tau_mats[0], a_sp)
    na_recs = []
    for ep in [0.2, 0.4, 0.6, 0.8, 1.0]:
        es = ym_e(Y0 + ep * A1s + ep * A2s) - ym_e(Y0 + ep * A1s) - ym_e(Y0 + ep * A2s) + ym_e(Y0)
        eu = ym_e(Y0 + ep * A1u + ep * A2u) - ym_e(Y0 + ep * A1u) - ym_e(Y0 + ep * A2u) + ym_e(Y0)
        na_recs.append({"eps": ep, "E_int_SU2": es, "E_int_U1": eu, "ratio_eps4": es / (ep ** 4)})

    max_higgs_err = max(r["err"] for r in sweep)
    max_ph_drift = max(abs(r["photon_shift"]) for r in sweep)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), dpi=140)
    fig.suptitle(f"Genesis-Rigorous v1.0.1 — Phase 3C (N={N}=2x{n_sphere}): Coulomb-Branch U(2) Matrix Higgs & Non-Abelian Coupling", fontsize=13, fontweight="bold", y=0.98)
    vs = [r["vev"] for r in sweep]
    axes[0].plot(vs, [r["W_shift"] for r in sweep], "ro", ms=7.5, label=f"Off-Diag W+- d(om^2) (Err={max_higgs_err:.1e})")
    axes[0].plot(vs, [r["exact_g2_v2"] for r in sweep], "k--", lw=1.8, label="Exact Matrix Higgs g^2 v^2")
    axes[0].plot(vs, [r["photon_shift"] for r in sweep], "b-s", ms=5.5, label=f"Diag U(1)xU(1) Photons (Drift={max_ph_drift:.1e})")
    axes[0].set_title("1. Coulomb-Branch Higgs Mass^2 Splitting\n(External Brane Separation v; U(2) -> U(1)xU(1))")
    axes[0].set_xlabel("Brane Separation VEV v"); axes[0].set_ylabel("Mass-Squared Shift d(omega^2)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Panel 2: Mode-tracked branches (zero zigzag: explicit diagonal vs off-diagonal lowest & 1st KK branches)
    axes[1].plot(vs, [r["photon_om2"] for r in sweep], "b-o", lw=2.2, label="Diag U(1)xU(1) Lowest (om^2 = mu^2 = 1)")
    axes[1].plot(vs, [r["W_om2"] for r in sweep], "r-s", lw=2.2, label="Off-Diag W+- Lowest (mu^2 + g^2 v^2)")
    axes[1].plot(vs, [r["kk_diag_om2"] for r in sweep], "b--^", lw=1.6, alpha=0.75, label="Diag 1st KK Branch (Unshifted)")
    axes[1].plot(vs, [r["kk_off_om2"] for r in sweep], "r--v", lw=1.6, alpha=0.75, label="Off-Diag 1st KK Branch (+g^2 v^2)")
    axes[1].set_title("2. Tracked Gauge & KK Spectrum vs VEV v\n(4-Fold U(2) Coincident Degeneracy Splits Cleanly)")
    axes[1].set_xlabel("Brane Separation VEV v"); axes[1].set_ylabel("Physical Mode Frequency Squared omega^2")
    axes[1].legend(fontsize=7.8, loc="upper left"); axes[1].grid(True, alpha=0.3)

    axes[2].plot([r["eps"] for r in na_recs], [r["E_int_SU2"] for r in na_recs], "m-o", lw=2.3, label="Non-Abelian SU(2) (~ eps^4)")
    axes[2].plot([r["eps"] for r in na_recs], [r["E_int_U1"] for r in na_recs], "c--x", lw=2.0, label="Abelian U(1) Control (= 0)")
    axes[2].set_title("3. Non-Abelian Yang-Mills Quartic Self-Interaction\n(Non-Commuting SU(2) vs Abelian U(1) Control)")
    axes[2].set_xlabel("Gauge Field Amplitude eps"); axes[2].set_ylabel("Quartic Interaction Energy E_int")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(os.path.join(out_dir, "phase3c_gauge_dashboard.png"), dpi=150); plt.close(fig)

    ram_mb, vram_mb = guard.check_runtime()
    report = {
        "N": N, "timestamp": ts, "out_dir": out_dir, "coincident_multiplet_spread": mult_spread,
        "max_higgs_err": max_higgs_err, "max_photon_drift": max_ph_drift,
        "higgs_sweep": sweep, "nonabelian": na_recs,
        "stats": {"elapsed_sec": time.time() - t0, "ram_mb": ram_mb, "vram_mb": vram_mb},
    }
    with open(os.path.join(out_dir, "phase3c_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    guard.cleanup()
    return report
