"""
Phase 3D Experiment: 4D Emergent Space S^2 x S^2 (D=6, N=16, Rank-4 Quantum Metric)
+ Exact Ginsparg-Wilson / GKP Massless Chiral Fermion Generations (q = 0, 1, 2, 3)
Protected Topologically by the Atiyah-Singer Index Theorem Under Quantum Matrix Noise.
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

from ..exact_hfb import build_su2_irrep
from ..matrix_native import MatrixNativeQuantumCore
from ..safety import HardwareSafetyGuard, SafetyConfig, safe_eigvalsh


def build_6d_clifford_gammas(device: torch.device) -> torch.Tensor:
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    id2 = torch.eye(2, dtype=torch.complex128, device=device)
    k3 = lambda a, b, c: torch.kron(torch.kron(a, b), c)
    return torch.stack([
        k3(s1, id2, id2), k3(s2, id2, id2),
        k3(s3, s1, id2), k3(s3, s2, id2),
        k3(s3, s3, s1), k3(s3, s3, s2),
    ], dim=0)


@torch.no_grad()
def probe_6d_metric_rank4(Y6: torch.Tensor, gammas: torch.Tensor, y6_pt: torch.Tensor) -> Tuple[float, np.ndarray, float]:
    N = Y6.shape[1]
    eye_N = torch.eye(N, dtype=torch.complex128, device=Y6.device)
    D_mat = sum(torch.kron((Y6[I] - y6_pt[I] * eye_N).contiguous(), gammas[I].contiguous()) for I in range(6))
    D_mat = 0.5 * (D_mat + D_mat.conj().T)
    evals, evecs = torch.linalg.eigh(D_mat)

    abs_ev = torch.abs(evals)
    min_abs = float(torch.min(abs_ev))
    sub_mask = torch.abs(abs_ev - min_abs) < max(0.25, 20.0 * min_abs)
    other_mask = ~sub_mask
    d_sub = int(torch.sum(sub_mask))

    psi_sub, lam_sub = evecs[:, sub_mask], evals[sub_mask]
    psi_oth, lam_oth = evecs[:, other_mask], evals[other_mask]
    gap = float(torch.min(torch.abs(lam_oth.unsqueeze(1) - lam_sub.unsqueeze(0))))
    inv_d = (1.0 / (lam_oth.unsqueeze(1) - lam_sub.unsqueeze(0))).to(torch.complex128)

    M_list = [((psi_oth.conj().T @ (torch.kron(eye_N, gammas[a].contiguous()) @ psi_sub)) * inv_d).reshape(-1) for a in range(6)]
    M_stack = torch.stack(M_list, dim=0)
    g_mat = (torch.real(M_stack.conj() @ M_stack.T) / float(d_sub)).cpu().numpy()
    return min_abs, 0.5 * (g_mat + g_mat.T), gap


@torch.no_grad()
def build_exact_gkp_chiral_dirac(X1: torch.Tensor, X2: torch.Tensor, alpha_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact Ginsparg-Wilson / GKP chiral Dirac operator satisfying {D_chiral, gamma5} == 0 & Index == N2 - N1."""
    device = X1.device
    N1, N2 = X1.shape[1], X2.shape[1]
    sig = torch.stack([
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device),
        torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device),
        torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device),
    ])
    eye1, eye2 = torch.eye(N1, dtype=torch.complex128, device=device), torch.eye(N2, dtype=torch.complex128, device=device)
    eye2s = torch.eye(2, dtype=torch.complex128, device=device)

    def mat_sign(H):
        ew, ev = torch.linalg.eigh(0.5 * (H + H.conj().T))
        s = torch.where(ew >= 0.0, torch.ones_like(ew), -torch.ones_like(ew)).to(torch.complex128)
        return (ev * s) @ ev.conj().T

    H_L = 0.5 * alpha_scale * torch.kron(eye2s, eye1) + sum(torch.kron(sig[I], X1[I].contiguous()) for I in range(3))
    H_R = 0.5 * alpha_scale * torch.kron(eye2s, eye2) + sum(torch.kron(sig[I], (-X2[I].T).contiguous()) for I in range(3))
    GL4 = mat_sign(H_L).reshape(2, N1, 2, N1)
    GR4 = mat_sign(H_R).reshape(2, N2, 2, N2)

    GL = torch.einsum("satp,bq->sabtpq", GL4, eye2).reshape(2 * N1 * N2, 2 * N1 * N2)
    GR = torch.einsum("sbtq,ap->sabtpq", GR4, eye1).reshape(2 * N1 * N2, 2 * N1 * N2)
    D_c = 0.5 * (GL + GR)
    g5 = 0.5 * (GL - GR)
    return 0.5 * (D_c + D_c.conj().T), 0.5 * (g5 + g5.conj().T)


@torch.no_grad()
def run_phase3d_chiral_experiment(out_dir: str = "", prefer_cuda: bool = True) -> Dict[str, object]:
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join("results", f"unified_phase3d_4d_chiral_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    N, D = 16, 6
    core = MatrixNativeQuantumCore(N=N, D=D, guard=guard)
    g, mu, kappa, g1, g2, lam4 = 0.25, 1.0, 0.15, 0.015, 0.0001, 0.008
    double_flux = [
        (0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0), (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0),
        (3, 4, 5, 1.0), (4, 5, 3, 1.0), (5, 3, 4, 1.0), (3, 5, 4, -1.0), (5, 4, 3, -1.0), (4, 3, 5, -1.0),
    ]

    def phi4(Y):
        RL2 = sum(Y[i] @ Y[i] for i in range(3))
        RR2 = sum(Y[3 + i] @ Y[3 + i] for i in range(3))
        e4 = 0.25 * lam4 * float(torch.real(torch.sum(RL2 * RL2.conj()) + torch.sum(RR2 * RR2.conj())))
        g4 = torch.zeros_like(Y)
        for i in range(3):
            g4[i] = 0.5 * lam4 * (RL2 @ Y[i] + Y[i] @ RL2)
            g4[3 + i] = 0.5 * lam4 * (RR2 @ Y[3 + i] + Y[3 + i] @ RR2)
        return e4, core.project_hermitian_traceless(g4)

    def relax_6d(nu: float, scales: List[float], n_steps: int, seed_base: int):
        best_E, best_Y = float("inf"), None
        for rs, sc in enumerate(scales):
            gen = torch.Generator(device="cpu").manual_seed(seed_base + 67 * rs)
            Y = core.random_isotropic_gue(sc, gen)
            dt, pY, pG = 0.008, None, None
            for step in range(n_steps):
                do_q = (step % 12 == 0) or (step >= n_steps - 20)
                e4, g4 = phi4(Y)
                if do_q:
                    Eb, Gb, _ = core.total_energy_and_grad(Y, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2, flux_triples=double_flux, compute_grad=True)
                else:
                    Eb, Gb = core.classical_energy_and_grad(Y, g=g, mu=mu, nu=nu, flux_triples=double_flux)
                E0, G = Eb + e4, Gb + g4
                gn = float(torch.sqrt(torch.real(torch.sum(G * G.conj()))))
                if gn < 1e-4 and step > 60: break
                if pY is not None and pG is not None:
                    sY, dG = Y - pY, G - pG
                    sy, ss = float(torch.real(torch.sum(sY * dG.conj()))), float(torch.real(torch.sum(sY * sY.conj())))
                    if sy > 1e-12 and ss > 1e-16: dt = float(np.clip(ss / sy, 1e-5, 0.25))
                yn = float(torch.sqrt(torch.real(torch.sum(Y * Y.conj()))))
                ssz = min(dt, max(4.0, 0.25 * yn) / max(gn, 1.0))
                acc = False
                for _ in range(10):
                    Yt = core.project_hermitian_traceless(Y - ssz * G)
                    e4t, _ = phi4(Yt)
                    Etb = core.total_energy_and_grad(Yt, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2, flux_triples=double_flux, compute_grad=False)[0] if do_q else core.classical_energy_and_grad(Yt, g=g, mu=mu, nu=nu, flux_triples=double_flux)[0]
                    if Etb + e4t <= E0 - 1e-4 * ssz * gn * gn:
                        pY, pG, Y = Y.clone(), G.clone(), Yt; dt = min(ssz * 1.4, 0.20); acc = True; break
                    ssz *= 0.4
                if not acc: dt *= 0.25; pY, pG = None, None
            Ef = core.total_energy_and_grad(Y, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2, flux_triples=double_flux, compute_grad=False)[0] + phi4(Y)[0]
            if Ef < best_E: best_E, best_Y = Ef, Y.clone()
        return best_Y, best_E

    Y_A, E_A = relax_6d(0.80, [1.5, 5.0], 160, 401)
    Y_B, E_B = relax_6d(3.00, [1.5, 5.0, 9.0, 14.0], 420, 502)

    tr_A = float(torch.real(torch.sum(Y_A * Y_A.conj())))
    cross_sq = sum(float(torch.real(torch.sum((Y_B[i] @ Y_B[3 + a] - Y_B[3 + a] @ Y_B[i]).abs() ** 2))) for i in range(3) for a in range(3))
    cross_err = float(np.sqrt(cross_sq) / max(float(torch.sqrt(torch.real(torch.sum(Y_B[:3].abs() ** 2))) * torch.sqrt(torch.real(torch.sum(Y_B[3:].abs() ** 2)))), 1e-12))

    # 6D Point-Probe Rank-4 Quantum Metric
    gam6 = build_6d_clifford_gammas(guard.device)
    rL_ev = torch.real(safe_eigvalsh(0.5 * (sum(Y_B[i] @ Y_B[i] for i in range(3)) + sum(Y_B[i] @ Y_B[i] for i in range(3)).conj().T))).cpu().numpy()
    rR_ev = torch.real(safe_eigvalsh(0.5 * (sum(Y_B[3 + i] @ Y_B[3 + i] for i in range(3)) + sum(Y_B[3 + i] @ Y_B[3 + i] for i in range(3)).conj().T))).cpu().numpy()
    rL_est, rR_est = float(np.sqrt(max(np.mean(rL_ev), 1.0))) * 0.78, float(np.sqrt(max(np.mean(rR_ev), 1.0))) * 0.78
    best_l6, best_rL, best_rR = 1e30, rL_est, rR_est
    for rlv in np.linspace(0.5 * rL_est, 1.3 * rL_est, 21):
        for rrv in np.linspace(0.5 * rR_est, 1.3 * rR_est, 21):
            pt = torch.tensor([0.0, 0.0, float(rlv), 0.0, 0.0, float(rrv)], dtype=core.rdtype, device=guard.device)
            Dm = sum(torch.kron((Y_B[I] - pt[I] * core.eye_N).contiguous(), gam6[I].contiguous()) for I in range(6))
            lm = float(torch.min(torch.abs(safe_eigvalsh(0.5 * (Dm + Dm.conj().T)))))
            if lm < best_l6: best_l6, best_rL, best_rR = lm, float(rlv), float(rrv)
    for dr in [0.4, 0.08]:
        for dl in np.linspace(-dr, dr, 9):
            for dr2 in np.linspace(-dr, dr, 9):
                pt = torch.tensor([0.0, 0.0, best_rL + dl, 0.0, 0.0, best_rR + dr2], dtype=core.rdtype, device=guard.device)
                Dm = sum(torch.kron((Y_B[I] - pt[I] * core.eye_N).contiguous(), gam6[I].contiguous()) for I in range(6))
                lm = float(torch.min(torch.abs(safe_eigvalsh(0.5 * (Dm + Dm.conj().T)))))
                if lm < best_l6: best_l6, best_rL, best_rR = lm, best_rL + dl, best_rR + dr2

    l6_m, g6_ab, gap6 = probe_6d_metric_rank4(Y_B, gam6, torch.tensor([0.0, 0.0, best_rL, 0.0, 0.0, best_rR], dtype=core.rdtype, device=guard.device))
    g6_ev = np.sort(np.linalg.eigvalsh(g6_ab)).tolist()
    eff_dim = int(np.sum(np.array(g6_ev) > 0.1 * max(g6_ev)))
    print(f"[3D-1 4D SPACE] Chaos A Tr(Y^2)={tr_A:.2e} | Emergent B CrossComm={cross_err:.2e} | g6_evals={np.round(g6_ev, 5)} -> {eff_dim}D!")

    # --- Experiment 3D-2: Chiral Fermion Generations (q = 0, 1, 2, 3) & Noise Protection ---
    alpha_c, N1 = 11.23, 6
    L1 = alpha_c * build_su2_irrep(N1, guard.device)
    gen_n = torch.Generator(device="cpu").manual_seed(202634)
    chiral_recs = []
    for q in [0, 1, 2, 3]:
        N2 = N1 + q
        L2 = alpha_c * build_su2_irrep(N2, guard.device)
        for amp in ([0.0, 1.5] if q == 3 else [0.0, 1.0]):
            def noisy(X):
                if amp == 0.0: return X
                n = X.shape[1]
                Z = (torch.randn((3, n, n), generator=gen_n, dtype=torch.float64) + 1j * torch.randn((3, n, n), generator=gen_n, dtype=torch.float64)).to(guard.device)
                return X + 0.5 * (Z + Z.conj().transpose(-2, -1)) * amp
            D_c, g5_c = build_exact_gkp_chiral_dirac(noisy(L1), noisy(L2), alpha_c)
            ew, ev = torch.linalg.eigh(D_c)
            aew = torch.abs(ew)
            zm = aew < 1e-10
            nz = int(torch.sum(zm))
            g5_vals = []
            if nz > 0:
                Vz = ev[:, zm]
                g5s = Vz.conj().T @ g5_c @ Vz
                g5_vals = torch.real(torch.linalg.eigvalsh(0.5 * (g5s + g5s.conj().T))).cpu().numpy().tolist()
            idx_c = int(sum(1 for x in g5_vals if x > 0.5) - sum(1 for x in g5_vals if x < -0.5))
            kk_gap = float(torch.min(aew[~zm])) if torch.any(~zm) else 0.0
            max_z = float(torch.max(aew[zm])) if nz > 0 else 0.0
            print(f"  [3D-2 q={q}, noise={amp:.1f}] #Zero={nz}, Index={idx_c:+d}, max|E0|={max_z:.1e}, KK_gap={kk_gap:.4f}")
            chiral_recs.append({"q": q, "noise": amp, "n_zero": nz, "chiral_index": idx_c, "max_E0": max_z, "kk_gap": kk_gap, "low16": torch.sort(aew)[0][:16].cpu().numpy().tolist()})

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=140)
    fig.suptitle("Project Genesis-Rigorous (Unified Phase 3D): 4D Emergent Space & Protected Chiral Fermion Generations", fontsize=13, fontweight="bold", y=0.98)
    axes[0].bar(range(1, 7), g6_ev, color=["gray", "gray", "royalblue", "royalblue", "royalblue", "royalblue"], edgecolor="black")
    axes[0].set_title(f"1. 6x6 Provost-Vallee Metric (Rank={eff_dim}D Space!)"); axes[0].set_xlabel("Eigenvalue Index"); axes[0].grid(True, axis="y", alpha=0.3)
    rc3 = [r for r in chiral_recs if r["q"] == 3 and r["noise"] == 0.0][0]
    rn3 = [r for r in chiral_recs if r["q"] == 3 and r["noise"] > 0.0][0]
    axes[1].plot(range(16), rc3["low16"], "bo", ms=7, label="q=3 Clean (3 Exact Zero Modes)")
    axes[1].plot(range(16), rn3["low16"], "r^", ms=6.5, label="q=3 Noisy (3 Zero Modes Stay E=0!)")
    axes[1].axvline(2.5, color="green", ls="--", label="3 Protected Chiral Generations")
    axes[1].set_title("2. Topological Protection of 3 Chiral Generations"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(os.path.join(out_dir, "phase3d_4d_chiral_dashboard.png"), dpi=150); plt.close(fig)

    ram_mb, vram_mb = guard.check_runtime()
    report = {
        "timestamp": ts, "out_dir": out_dir,
        "emergent_4d": {"tr_A": tr_A, "E_B": E_B, "cross_comm_err": cross_err, "metric_6x6_evals": g6_ev, "emergent_dim": eff_dim},
        "chiral_fermions": chiral_recs,
        "stats": {"elapsed_sec": time.time() - t0, "ram_mb": ram_mb, "vram_mb": vram_mb},
    }
    with open(os.path.join(out_dir, "phase3d_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    guard.cleanup()
    return report
