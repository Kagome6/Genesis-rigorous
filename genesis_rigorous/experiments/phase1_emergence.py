"""
Phase 1 Experiment: Dual-Universe Quantum Phase Transition (Stage 1)
+ Point-Probe Dirac Geometry & Exact Provost-Vallee Metric (Stage 2A)
+ Blind Minimal-Area Entanglement Search (Stage 2B).
"""

import datetime
import json
import os
import time
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .. import safety as safety_mod
from ..exact_hfb import GenesisQuantumCore, analyze_matrix_geometry, build_hermitian_basis
from ..safety import HardwareSafetyGuard, SafetyConfig, safe_eigvalsh


@torch.no_grad()
def run_stage1_universe(
    core: GenesisQuantumCore, label: str, nu: float,
    g: float = 0.25, mu: float = 1.0, kappa: float = 0.15, Nf: int = 1,
    g1: float = 0.015, g2: float = 0.0001,
    n_steps: int = 240, n_restarts: int = 4, hfb_interval: int = 35,
    temp_init: float = 0.30, seed_base: int = 100
) -> Dict[str, object]:
    scale_pool = [1.0, 6.0, 15.0, 22.0]
    print("-" * 92)
    print(f"[STAGE 1] Running {label} | N={core.N}, nu={nu:.2f}, g={g:.2f}, mu={mu:.2f}, kappa={kappa:.2f}, restarts={n_restarts}")

    best_E = float("inf")
    best_bundle = None
    restart_summaries = []

    def fast_total_y_energy(y_v: torch.Tensor, Sx_t: torch.Tensor) -> float:
        e_y = core.mqm_y_energy_fixed_sx(y_v, Sx_t, g=g, mu=mu, nu=nu)
        y_r = y_v.reshape(core.D, core.N2)
        F_m = (y_r @ core.f_flat).reshape(core.D, core.N2, core.N2)
        Delta = sum(F_m[I] @ F_m[I].T for I in range(core.D))
        lam = torch.clamp(safe_eigvalsh(0.5 * (Delta + Delta.T)), min=0.0)
        f_v = g1 * lam + g2 * (lam * lam)
        e_tn = float(kappa * Nf * torch.sum(f_v - torch.sqrt(f_v * f_v + 1.0)))
        return e_y + e_tn

    for rs in range(n_restarts):
        if safety_mod.STOP_REQUESTED:
            break

        y_scale = scale_pool[rs % len(scale_pool)]
        gen = torch.Generator(device="cpu").manual_seed(seed_base + 53 * rs)
        y = (torch.randn(core.nB, generator=gen, dtype=core.rdtype) * y_scale).to(core.device)
        y = core.project_traceless(y)

        Omega, Sx, _, _ = core.solve_hfb_omega(y, g=g, mu=mu, nu=nu, max_iter=8)
        dt = 0.008
        prev_y, prev_g = None, None
        hist = {"steps": [], "E_tot": [], "E_mqm": [], "E_tn": [], "tr_Y2": [], "alg_err": []}

        for step in range(n_steps + 1):
            if safety_mod.STOP_REQUESTED:
                break
            if step > 0 and step % hfb_interval == 0:
                Omega, Sx, _, _ = core.solve_hfb_omega(y, g=g, mu=mu, nu=nu, max_iter=6, Omega_init=Omega)
                core.guard.check_runtime()
                prev_y, prev_g = None, None

            grad, E_tn = core.exact_grad_y(y, Sx, g=g, mu=mu, nu=nu, kappa=kappa, Nf=Nf, g1=g1, g2=g2)

            if step % 25 == 0 or step == n_steps:
                e_dict = core.mqm_quantum_energy(y, Omega, Sx, g=g, mu=mu, nu=nu)
                diag = analyze_matrix_geometry(core.vec_to_mats(y))
                hist["steps"].append(step)
                hist["E_tot"].append(e_dict["E_mqm"] + E_tn)
                hist["E_mqm"].append(e_dict["E_mqm"])
                hist["E_tn"].append(E_tn)
                hist["tr_Y2"].append(diag["tr_Y2"])
                hist["alg_err"].append(diag["rel_alg_err"])

            if step == n_steps:
                break

            gn = float(torch.linalg.norm(grad))
            if gn < 1e-6 and step > int(0.65 * n_steps):
                continue

            if prev_y is not None and prev_g is not None:
                s_v, d_g = y - prev_y, grad - prev_g
                sy, ss = float(torch.dot(s_v, d_g)), float(torch.dot(s_v, s_v))
                if sy > 1e-12 and ss > 1e-16:
                    dt = float(np.clip(ss / sy, 1e-5, 0.25))

            E0 = fast_total_y_energy(y, Sx)
            step_sz = min(dt, 4.0 / max(gn, 1.0))
            accepted = False
            for _ in range(12):
                y_try = core.project_traceless(y - step_sz * grad)
                if fast_total_y_energy(y_try, Sx) <= E0 - 1e-4 * step_sz * (gn * gn):
                    prev_y, prev_g = y.clone(), grad.clone()
                    y = y_try
                    dt = min(step_sz * 1.4, 0.20)
                    accepted = True
                    break
                step_sz *= 0.4
            if not accepted:
                dt *= 0.25
                prev_y, prev_g = None, None

            cool_frac = max(0.0, 1.0 - step / (0.45 * n_steps))
            temp = temp_init * (cool_frac ** 2)
            if temp > 0.0:
                noise = (torch.randn(core.nB, generator=gen, dtype=core.rdtype) * np.sqrt(2.0 * temp * min(dt, 0.02))).to(core.device)
                y = core.project_traceless(y + noise)

        for _ in range(2):
            Omega, Sx, _, _ = core.solve_hfb_omega(y, g=g, mu=mu, nu=nu, max_iter=15, Omega_init=Omega)
            for _ in range(25):
                grad, _ = core.exact_grad_y(y, Sx, g=g, mu=mu, nu=nu, kappa=kappa, Nf=Nf, g1=g1, g2=g2)
                gn = float(torch.linalg.norm(grad))
                if gn < 1e-6:
                    break
                E0 = fast_total_y_energy(y, Sx)
                step_sz = min(0.05, 2.0 / max(gn, 1.0))
                for _ in range(10):
                    y_try = core.project_traceless(y - step_sz * grad)
                    if fast_total_y_energy(y_try, Sx) <= E0 - 1e-4 * step_sz * (gn * gn):
                        y = y_try
                        break
                    step_sz *= 0.4

        Omega, Sx, _, _ = core.solve_hfb_omega(y, g=g, mu=mu, nu=nu, max_iter=18, Omega_init=Omega)
        grad, E_tn = core.exact_grad_y(y, Sx, g=g, mu=mu, nu=nu, kappa=kappa, Nf=Nf, g1=g1, g2=g2)
        e_dict = core.mqm_quantum_energy(y, Omega, Sx, g=g, mu=mu, nu=nu)
        E_tot = e_dict["E_mqm"] + E_tn
        Y_f = core.vec_to_mats(y)
        diag = analyze_matrix_geometry(Y_f)
        print(
            f"  [Restart {rs+1}/{n_restarts} (scale={y_scale:4.1f})] E_tot={E_tot:10.3f} (MQM={e_dict['E_mqm']:10.3f}, TN={E_tn:7.3f}) "
            f"| Tr(Y^2)={diag['tr_Y2']:9.2f} | AlgErr={diag['rel_alg_err']:.3e} | ||grad||={float(torch.linalg.norm(grad)):.2e}"
        )
        restart_summaries.append({
            "restart": rs + 1, "y_scale": y_scale, "E_tot": E_tot,
            "tr_Y2": diag["tr_Y2"], "rel_alg_err": diag["rel_alg_err"],
            "r2_evals": [round(v, 2) for v in diag["r2_evals"]],
        })
        if E_tot < best_E:
            best_E = E_tot
            best_bundle = {
                "label": label, "nu": nu, "y": y.clone(), "Y": Y_f.clone(),
                "Omega": Omega.clone(), "Sx": Sx.clone(),
                "E_tot": E_tot, "E_mqm": e_dict["E_mqm"], "E_tn": E_tn,
                "grad_norm": float(torch.linalg.norm(grad)),
                "diag": diag, "history": hist, "restarts": restart_summaries,
            }

    best_bundle["restarts"] = restart_summaries
    d = best_bundle["diag"]
    print(f"  >>> SELECTED LOWEST ENERGY: E_tot={best_bundle['E_tot']:.3f} | Tr(Y^2)={d['tr_Y2']:.4f} | R_rms={d['R_rms']:.4f} | AlgErr={d['rel_alg_err']:.3e}")
    return best_bundle


@torch.no_grad()
def run_stage2a_point_probe(core: GenesisQuantumCore, Y: torch.Tensor, R_ref: float, grid_n: int = 37) -> Dict[str, object]:
    lim = max(1.5 * R_ref, 2.0)
    xs = np.linspace(-lim, lim, grid_n)
    slice_2d = np.zeros((grid_n, grid_n), dtype=float)
    for ix, xv in enumerate(xs):
        for iy, yv in enumerate(xs):
            pt = torch.tensor([xv, yv, 0.0], dtype=core.rdtype, device=core.device)
            slice_2d[iy, ix], _ = core.point_probe_dirac(Y, pt)

    r_vals = np.linspace(0.0, lim, 101)
    radial_lmin = np.array([
        core.point_probe_dirac(Y, torch.tensor([0.0, 0.0, float(rv)], dtype=core.rdtype, device=core.device))[0]
        for rv in r_vals
    ], dtype=float)

    idx_min = int(np.argmin(radial_lmin))
    r_best, lmin_best = float(r_vals[idx_min]), float(radial_lmin[idx_min])
    if 0 < idx_min < len(r_vals) - 1:
        a_r, b_r = float(r_vals[idx_min - 1]), float(r_vals[idx_min + 1])
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        for _ in range(18):
            c_r, d_r = b_r - gr * (b_r - a_r), a_r + gr * (b_r - a_r)
            lc = core.point_probe_dirac(Y, torch.tensor([0.0, 0.0, c_r], dtype=core.rdtype, device=core.device))[0]
            ld = core.point_probe_dirac(Y, torch.tensor([0.0, 0.0, d_r], dtype=core.rdtype, device=core.device))[0]
            if lc < ld: b_r = d_r
            else: a_r = c_r
        r_ref_opt = 0.5 * (a_r + b_r)
        l_ref_opt = core.point_probe_dirac(Y, torch.tensor([0.0, 0.0, r_ref_opt], dtype=core.rdtype, device=core.device))[0]
        if l_ref_opt <= lmin_best:
            r_best, lmin_best = r_ref_opt, l_ref_opt

    y_best = torch.tensor([0.0, 0.0, max(r_best, 0.05)], dtype=core.rdtype, device=core.device)
    g_ab, g_gap = core.provost_vallee_metric_exact(Y, y_best)
    return {
        "lim": lim, "xs": xs, "slice_2d": slice_2d,
        "r_vals": r_vals, "radial_lmin": radial_lmin,
        "r_best": r_best, "lmin_at_r_best": lmin_best,
        "provost_vallee_evals": np.linalg.eigvalsh(g_ab).tolist(),
        "provost_vallee_gap": g_gap,
    }


def run_stage2b_blind_qrf(
    core: GenesisQuantumCore, bundle: Dict[str, object],
    kappa: float = 0.15, Nf: int = 1, g1: float = 0.015, g2: float = 0.0001,
    opt_steps: int = 100, seed: int = 777
) -> Dict[str, object]:
    Y, Omega = bundle["Y"], bundle["Omega"]
    R_rms = max(bundle["diag"]["R_rms"], 1e-12)

    with torch.no_grad():
        _, lam, f, E_mats, _ = core.bdg_link_sector(Y, kappa=kappa, Nf=Nf, g1=g1, g2=g2)
        cache = core.precompute_boson_entanglement_cache(Omega, Y=Y, project_gauss=True)
        gens = core.T_basis[1:].detach()
        K_use = gens.shape[0]

    @torch.no_grad()
    def geom_metrics(Q_in: torch.Tensor, M_rank: int) -> Tuple[float, float]:
        Th = Q_in @ Q_in.conj().T
        cm = torch.stack([torch.real(torch.trace(Th @ Y[I])) / float(M_rank) for I in range(3)])
        pol = float(torch.linalg.norm(cm)) / R_rms
        Th_bar = core.eye_N - Th
        B_m = sum((Th @ Y[I] @ Th_bar) @ (Th_bar @ Y[I] @ Th) for I in range(3))
        ev_b = torch.clamp(torch.real(safe_eigvalsh(0.5 * (B_m + B_m.conj().T))), min=0.0)
        return pol, float(np.sqrt(2.0) * torch.sum(torch.sqrt(ev_b))) / R_rms

    results_M, traj_mid, Q_opt_mid = [], None, None
    M_mid = core.N // 2
    gen_q = torch.Generator(device="cpu").manual_seed(seed)
    print(f"\n[STAGE 2B] Blind QRF Minimal-Area Scan ({bundle['label']}, full_su({core.N}))")
    print("  M | S_EE: Init ->  Min   | S_TN   | S_bos  | Cas_U(N) | Indep.Pol: Init->Opt | Indep.Perim: Init->Opt")

    for M_rank in range(1, core.N):
        if safety_mod.STOP_REQUESTED:
            break
        Z_q = (torch.randn((core.N, core.N), generator=gen_q, dtype=core.rdtype)
               + 1j * torch.randn((core.N, core.N), generator=gen_q, dtype=core.rdtype)).to(core.device)
        U_0, _ = torch.linalg.qr(Z_q)
        Q_0 = U_0[:, :M_rank].detach()
        T_M = build_hermitian_basis(M_rank, device=core.device)

        c_param = torch.zeros(K_use, dtype=core.rdtype, device=core.device, requires_grad=True)
        optimizer = torch.optim.Adam([c_param], lr=0.14)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_steps, eta_min=0.015)

        pol_init, perim_init = geom_metrics(Q_0, M_rank)
        obj_init = None
        local_traj = {"step": [], "obj": [], "perim": [], "pol": []}

        for st in range(opt_steps):
            optimizer.zero_grad()
            H_g = torch.einsum("k,kij->ij", c_param.to(core.cdtype), gens)
            H_g = 0.5 * (H_g + H_g.conj().T)
            Q_in = torch.linalg.matrix_exp(1j * H_g) @ Q_0
            S_tn = core.bdg_subregion_entropy(E_mats, f, Q_in, Nf=Nf)
            _, S_bos = core.gauss_casimir_and_boson_entropy(Y, cache, Q_in, T_M=T_M, full_casimir=False)
            loss = S_tn + S_bos
            if obj_init is None:
                obj_init = float(loss.detach())
            if M_rank == M_mid and (st % 4 == 0 or st == opt_steps - 1):
                pc, ac = geom_metrics(Q_in.detach(), M_rank)
                local_traj["step"].append(st); local_traj["obj"].append(float(loss.detach()))
                local_traj["perim"].append(ac); local_traj["pol"].append(pc)
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            H_g = torch.einsum("k,kij->ij", c_param.to(core.cdtype), gens)
            H_g = 0.5 * (H_g + H_g.conj().T)
            Q_opt = torch.linalg.matrix_exp(1j * H_g) @ Q_0
            S_tn_o = float(core.bdg_subregion_entropy(E_mats, f, Q_opt, Nf=Nf))
            cas_full, S_bos_o = core.gauss_casimir_and_boson_entropy(Y, cache, Q_opt, T_M=T_M, full_casimir=True)
            cas_o, S_bos_f = float(cas_full), float(S_bos_o)
            obj_opt = S_tn_o + S_bos_f
            pol_opt, perim_opt = geom_metrics(Q_opt, M_rank)

        print(
            f"  {M_rank:2d} | {obj_init:7.3f} -> {obj_opt:7.3f} | {S_tn_o:6.3f} | {S_bos_f:6.3f} | {cas_o:8.1f} | "
            f"   {pol_init:.3f} -> {pol_opt:.3f}    |    {perim_init:.3f} -> {perim_opt:.3f}"
        )
        results_M.append({
            "M": M_rank, "obj_init": obj_init, "obj_opt": obj_opt,
            "S_tn": S_tn_o, "S_bos": S_bos_f, "casimir": cas_o,
            "pol_init": pol_init, "pol_opt": pol_opt,
            "perim_init": perim_init, "perim_opt": perim_opt,
        })
        if M_rank == M_mid:
            traj_mid = local_traj
            Q_opt_mid = Q_opt.detach().cpu()

    Ms = np.array([r["M"] for r in results_M], dtype=float)
    cap_shape = np.sqrt(Ms * (core.N - Ms))
    s_raw = np.array([r["obj_opt"] for r in results_M], dtype=float)
    s_sym = 0.5 * (s_raw + s_raw[::-1])
    s_tn_arr = np.array([r["S_tn"] for r in results_M], dtype=float)
    perim_arr = np.array([r["perim_opt"] for r in results_M], dtype=float)

    def safe_corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0

    stats = {
        "corr_S_sym": safe_corr(s_sym, cap_shape),
        "corr_S_tn": safe_corr(s_tn_arr, cap_shape),
        "corr_perim": safe_corr(perim_arr, cap_shape),
        "mean_perim_shrink_pct": float(100.0 * np.mean([(r["perim_init"] - r["perim_opt"]) / max(r["perim_init"], 1e-12) for r in results_M])),
    }
    print(f"  >>> Area-Law Stats: r(S_sym)={stats['corr_S_sym']:.4f} | r(S_TN)={stats['corr_S_tn']:.4f} | Mean Perim Shrink={stats['mean_perim_shrink_pct']:+.1f}%")
    return {"laplacian_evals": lam.cpu().numpy().tolist(), "results_M": results_M, "traj_mid": traj_mid, "M_mid": M_mid, "Q_opt_mid": Q_opt_mid, "area_law_stats": stats}


def plot_phase1_dashboard(N: int, uni_A, uni_B, probe_A, probe_B, qrf_A, qrf_B, out_png: str):
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.8), dpi=140)
    fig.suptitle(f"Project Genesis-Rigorous (Unified Phase 1, N={N}) — Zero-Cheat Spacetime & Area-Law Emergence", fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    ax.plot(uni_A["history"]["steps"], uni_A["history"]["tr_Y2"], "r--o", lw=2.0, ms=4, label=f"Universe A (Chaos, nu={uni_A['nu']:.1f})")
    ax.plot(uni_B["history"]["steps"], uni_B["history"]["tr_Y2"], "b-s", lw=2.2, ms=4, label=f"Universe B (Emergent, nu={uni_B['nu']:.1f})")
    ax.set_title("1. Stage 1: Vacuum-Subtracted Geometry Tr(Y^2)", fontsize=10); ax.set_xlabel("Step"); ax.set_ylabel("Tr(Y^2)")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    lim = probe_B["lim"]
    im = ax.imshow(probe_B["slice_2d"], origin="lower", extent=[-lim, lim, -lim, lim], cmap="inferno_r", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="min |lambda(D_probe)|")
    ax.set_title(f"2. Stage 2A: Point-Probe Slice z=0 (Bright Ring = Manifold, r={probe_B['r_best']:.2f})", fontsize=9.5)

    ax = axes[0, 2]
    ax.plot(probe_A["r_vals"], probe_A["radial_lmin"], "r--", lw=2.0, label="Universe A (No Shell)")
    ax.plot(probe_B["r_vals"], probe_B["radial_lmin"], "b-", lw=2.4, label=f"Universe B (r={probe_B['r_best']:.2f})")
    ax.set_title("3. Stage 2A: Radial Point-Probe Spectrum", fontsize=10); ax.set_xlabel("r"); ax.set_ylabel("|lambda_min|")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    trB = qrf_B["traj_mid"]
    if trB is not None:
        ax_t = ax.twinx()
        l1 = ax.plot(trB["step"], trB["obj"], "b-o", lw=2.0, ms=3, label="Blind Entanglement S_EE (Left)")
        l2 = ax_t.plot(trB["step"], trB["perim"], "g--^", lw=2.0, ms=4, label="Independent NC Perimeter (Right)")
        ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="upper right", fontsize=8)
    ax.set_title(f"4. Stage 2B: Blind Minimization Trajectory (M={qrf_B['M_mid']})", fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    Ms = np.array([r["M"] for r in qrf_B["results_M"]], dtype=float)
    sA = np.array([r["obj_opt"] for r in qrf_A["results_M"]], dtype=float)
    sB = np.array([r["obj_opt"] for r in qrf_B["results_M"]], dtype=float)
    sB_tn = np.array([r["S_tn"] for r in qrf_B["results_M"]], dtype=float)
    sA_sym, sB_sym = 0.5 * (sA + sA[::-1]), 0.5 * (sB + sB[::-1])
    cap = np.sqrt(Ms * (N - Ms))
    if np.max(cap) > 0 and np.max(sB_sym) > 0:
        cap = cap * (np.max(sB_sym) / np.max(cap))
    ax.plot(Ms, sA_sym, "r--x", lw=1.8, label="Universe A (Chaos)")
    ax.plot(Ms, sB, "c:s", lw=1.6, ms=4, label="Univ B Raw 4-Block S_EE(M) (Asym)")
    ax.plot(Ms, sB_sym, "b-o", lw=2.2, label=f"Univ B Sym S_bar(M) (r={qrf_B['area_law_stats']['corr_S_sym']:.3f})")
    ax.plot(Ms, sB_tn, "g-^", lw=1.8, ms=4.5, label=f"Univ B Raw S_TN(M) (r={qrf_B['area_law_stats']['corr_S_tn']:.3f})")
    ax.plot(Ms, cap, "k:", lw=1.6, label="2D Area-Law ~ sqrt(M(N-M))")
    ax.set_title("5. Stage 2B: Minimal Entanglement vs Subregion Rank M", fontsize=10); ax.set_xticks(Ms)
    ax.legend(loc="best", fontsize=7.0); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    lam_A, lam_B = np.array(qrf_A["laplacian_evals"]), np.array(qrf_B["laplacian_evals"])
    ks = min(30, len(lam_B))
    ax.plot(range(ks), lam_A[:ks], "r--x", lw=1.8, label="Universe A")
    ax.plot(range(ks), lam_B[:ks], "b-o", lw=2.0, label="Universe B (l(l+1) Multiplets)")
    ax.set_title("6. Emergent Noncommutative Laplacian Spectrum", fontsize=10); ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def run_phase1_experiment(
    N: int = 8, g: float = 0.25, mu: float = 1.0, nu_chaos: float = 0.80, nu_emerg: float = 3.00,
    kappa: float = 0.15, g1: float = 0.015, g2: float = 0.0001,
    steps: int = 240, restarts: int = 4, restarts_chaos: int = 2, qrf_steps: int = 100,
    out_dir: str = "", prefer_cuda: bool = True
) -> Dict[str, object]:
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join("results", f"unified_phase1_N{N}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    core = GenesisQuantumCore(N=N, D=3, guard=guard)

    uni_A = run_stage1_universe(core, "Universe A (Chaos Control)", nu=nu_chaos, g=g, mu=mu, kappa=kappa, g1=g1, g2=g2, n_steps=steps, n_restarts=restarts_chaos, seed_base=101)
    uni_B = run_stage1_universe(core, "Universe B (Emergent Phase)", nu=nu_emerg, g=g, mu=mu, kappa=kappa, g1=g1, g2=g2, n_steps=steps, n_restarts=restarts, seed_base=202)

    R_ref = max(uni_B["diag"]["R_rms"], 2.0)
    probe_A = run_stage2a_point_probe(core, uni_A["Y"], R_ref=R_ref)
    probe_B = run_stage2a_point_probe(core, uni_B["Y"], R_ref=R_ref)
    qrf_A = run_stage2b_blind_qrf(core, uni_A, kappa=kappa, g1=g1, g2=g2, opt_steps=qrf_steps, seed=701)
    qrf_B = run_stage2b_blind_qrf(core, uni_B, kappa=kappa, g1=g1, g2=g2, opt_steps=qrf_steps, seed=702)

    out_png = os.path.join(out_dir, "phase1_dashboard.png")
    plot_phase1_dashboard(N, uni_A, uni_B, probe_A, probe_B, qrf_A, qrf_B, out_png)

    ram_mb, vram_mb = guard.check_runtime()
    report = {
        "N": N, "timestamp": ts, "out_dir": out_dir, "device": str(core.device),
        "params": {"g": g, "mu": mu, "nu_chaos": nu_chaos, "nu_emerg": nu_emerg, "kappa": kappa, "g1": g1, "g2": g2},
        "universe_A": {"E_tot": uni_A["E_tot"], "diag": uni_A["diag"], "probe_r_best": probe_A["r_best"], "area_law_stats": qrf_A["area_law_stats"], "qrf": qrf_A["results_M"]},
        "universe_B": {"E_tot": uni_B["E_tot"], "diag": uni_B["diag"], "probe_r_best": probe_B["r_best"], "provost_vallee_evals": probe_B["provost_vallee_evals"], "area_law_stats": qrf_B["area_law_stats"], "qrf": qrf_B["results_M"]},
        "stats": {"elapsed_sec": time.time() - t0, "ram_mb": ram_mb, "vram_mb": vram_mb},
    }
    with open(os.path.join(out_dir, "phase1_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    out_pt = os.path.join(out_dir, "phase1_states.pt")
    torch.save({
        "N": N, "params": report["params"],
        "universe_A": {"y": uni_A["y"].cpu(), "Y": uni_A["Y"].cpu(), "Omega": uni_A["Omega"].cpu(), "Sx": uni_A["Sx"].cpu(), "E_tot": uni_A["E_tot"]},
        "universe_B": {"y": uni_B["y"].cpu(), "Y": uni_B["Y"].cpu(), "Omega": uni_B["Omega"].cpu(), "Sx": uni_B["Sx"].cpu(), "E_tot": uni_B["E_tot"]},
    }, out_pt)
    guard.cleanup()
    report["state_pt"] = out_pt
    return report
