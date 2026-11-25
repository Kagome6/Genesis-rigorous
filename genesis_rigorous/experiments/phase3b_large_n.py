"""
Phase 3B Experiment: O(N^3) Matrix-Native Large-N Continuum Limit (N = 16, 20, 24, 32).
Zero O(N^6) f_abc allocation -> runs N=32 in ~8 MB VRAM on RTX 3060!
"""

import datetime
import json
import os
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..dirac_matter import find_shell_radius_along_ray
from ..exact_hfb import analyze_matrix_geometry
from ..matrix_native import MatrixNativeQuantumCore
from ..safety import HardwareSafetyGuard, SafetyConfig


@torch.no_grad()
def relax_large_n_universe(
    core: MatrixNativeQuantumCore, nu: float, g: float, mu: float, kappa: float, g1: float, g2: float,
    scales: List[float], n_steps: int = 420, seed_base: int = 300
) -> Dict[str, object]:
    best_E, best_Y, best_res = float("inf"), None, None
    for rs, y_scale in enumerate(scales):
        gen = torch.Generator(device="cpu").manual_seed(seed_base + 73 * rs)
        Y = core.random_isotropic_gue(y_scale, gen)
        dt, prev_Y, prev_G = 0.008, None, None
        for step in range(n_steps):
            do_q = (step % 6 == 0) or (step >= n_steps - 25)
            if do_q:
                E0, G, _ = core.total_energy_and_grad(Y, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2, compute_grad=True)
            else:
                E0, G = core.classical_energy_and_grad(Y, g=g, mu=mu, nu=nu)
            gn = float(torch.sqrt(torch.real(torch.sum(G * G.conj()))))
            if gn < 1e-4 and step > 60:
                break
            if prev_Y is not None and prev_G is not None:
                sY, dG = Y - prev_Y, G - prev_G
                sy, ss = float(torch.real(torch.sum(sY * dG.conj()))), float(torch.real(torch.sum(sY * sY.conj())))
                if sy > 1e-12 and ss > 1e-16:
                    dt = float(np.clip(ss / sy, 1e-5, 0.25))
            y_norm = float(torch.sqrt(torch.real(torch.sum(Y * Y.conj()))))
            step_sz = min(dt, max(4.0, 0.25 * y_norm) / max(gn, 1.0))
            acc = False
            for _ in range(10):
                Yt = core.project_hermitian_traceless(Y - step_sz * G)
                Et = core.total_energy_and_grad(Yt, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2, compute_grad=False)[0] if do_q else core.classical_energy_and_grad(Yt, g=g, mu=mu, nu=nu)[0]
                if Et <= E0 - 1e-4 * step_sz * gn * gn:
                    prev_Y, prev_G, Y = Y.clone(), G.clone(), Yt
                    dt = min(step_sz * 1.4, 0.20); acc = True; break
                step_sz *= 0.4
            if not acc:
                dt *= 0.25; prev_Y, prev_G = None, None

        Ef, _, qf = core.total_energy_and_grad(Y, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2, compute_grad=False)
        if Ef < best_E:
            best_E, best_Y, best_res = Ef, Y.clone(), qf

    diag = analyze_matrix_geometry(best_Y)
    if diag["R_rms"] > 0.5:
        r_best, lmin_best = find_shell_radius_along_ray(core, best_Y, np.array([0.0, 0.0, 1.0]), r_hint=diag["R_rms"])
        g_ab, _ = core.provost_vallee_metric_exact(best_Y, torch.tensor([0.0, 0.0, r_best], dtype=core.rdtype, device=core.device))
        g_ev = np.sort(np.linalg.eigvalsh(g_ab)).tolist()
    else:
        r_best, lmin_best, g_ev = 0.0, 0.0, [0.0, 0.0, 0.0]

    return {
        "Y": best_Y, "E_tot": best_E, "diag": diag, "r_best": r_best, "lmin_best": lmin_best,
        "g_evals": g_ev, "laplacian_evals": best_res["lam"].cpu().numpy().tolist(),
    }


def run_phase3b_large_n_experiment(
    N_list: List[int] = None, out_dir: str = "", prefer_cuda: bool = True
) -> Dict[str, object]:
    if N_list is None:
        N_list = [16, 20, 24, 32]
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join("results", f"unified_phase3b_largeN_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    g, mu, nu_A, nu_B, kappa, g1, g2 = 0.25, 1.0, 0.80, 3.00, 0.15, 0.015, 0.0001

    records = []
    for N in N_list:
        tn = time.time()
        core = MatrixNativeQuantumCore(N=N, D=3, guard=guard)
        scales = [2.0, 18.0, 30.0, 45.0]
        uA = relax_large_n_universe(core, nu_A, g, mu, kappa, g1, g2, scales[:2], n_steps=160, seed_base=101 + N)
        uB = relax_large_n_universe(core, nu_B, g, mu, kappa, g1, g2, scales, n_steps=420, seed_base=202 + N)
        ram_mb, vram_mb = guard.check_runtime()
        print(
            f"[3B N={N:2d}] Chaos A: Tr(Y^2)={uA['diag']['tr_Y2']:5.2f} | Emergent B: E={uB['E_tot']:11.1f}, "
            f"alpha={uB['diag']['alpha_fit']:.4f}, AlgErr={uB['diag']['rel_alg_err']:.2e}, r_best={uB['r_best']:.2f}, "
            f"g_t={uB['g_evals'][1]:.6f} | {time.time()-tn:.1f}s | VRAM={vram_mb:.1f}MB"
        )
        records.append({
            "N": N, "elapsed_sec": time.time() - tn, "ram_mb": ram_mb, "vram_mb": vram_mb,
            "uni_A": {"E_tot": uA["E_tot"], "tr_Y2": uA["diag"]["tr_Y2"]},
            "uni_B": {"E_tot": uB["E_tot"], "diag": uB["diag"], "r_best": uB["r_best"], "g_evals": uB["g_evals"], "laplacian_first64": uB["laplacian_evals"][:64]},
        })
        del core
        guard.cleanup()

    # Plot summary figure
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=140)
    fig.suptitle(f"Project Genesis-Rigorous (Unified Phase 3B): O(N^3) Large-N Continuum Limit ({N_list})", fontsize=13.5, fontweight="bold", y=0.98)
    Ns = np.array([r["N"] for r in records], dtype=float)
    rB = np.array([r["uni_B"]["r_best"] for r in records], dtype=float)
    gT = np.array([0.5 * (r["uni_B"]["g_evals"][1] + r["uni_B"]["g_evals"][2]) for r in records], dtype=float)
    axes[0].plot(Ns, gT, "bo", ms=8, label="Measured Provost-Vallee g_tang(N)")
    axes[0].plot(Ns, (0.5 * Ns) / (2.0 * np.maximum(rB, 1e-6) ** 2), "r--", lw=2.0, label="Continuum Theory (N/2)/(2 r^2)")
    axes[0].set_title("1. Emergent Metric Scaling g_tang(N)"); axes[0].set_xlabel("N"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    rmax = records[-1]
    lam_n = np.array(rmax["uni_B"]["laplacian_first64"]) / max(rmax["uni_B"]["diag"]["alpha_fit"] ** 2, 1e-12)
    ex = [l * (l + 1) for l in range(8) for _ in range(2 * l + 1)][:len(lam_n)]
    axes[1].plot(range(len(lam_n)), lam_n, "bo", ms=4.5, label=f"Emergent Delta_Y / alpha^2 (N={rmax['N']})")
    axes[1].plot(range(len(lam_n)), ex, "r-", lw=1.6, label="Exact S^2 Harmonics l(l+1)")
    axes[1].set_title(f"2. Spherical Harmonics at N={rmax['N']}"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(os.path.join(out_dir, "phase3b_largeN_dashboard.png"), dpi=150); plt.close(fig)

    report = {"timestamp": ts, "out_dir": out_dir, "records": records, "elapsed_sec": time.time() - t0}
    with open(os.path.join(out_dir, "phase3b_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
