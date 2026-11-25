"""
Phase 2 / Stage 3 Experiment: Emergent Gravity from Localized Adjoint Dirac Matter.
Measures local Gaussian curvature K(theta), Provost-Vallee quantum metric response,
North-vs-South entanglement first law, and blind minimal-surface gravitational deflection.
"""

import datetime
import glob
import json
import os
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..dirac_matter import AdjointDiracMatterSector, find_shell_radius_along_ray, scan_meridian_geometry
from ..exact_hfb import GenesisQuantumCore, build_hermitian_basis
from ..safety import HardwareSafetyGuard, SafetyConfig, safe_eigh, safe_eigvalsh


@torch.no_grad()
def reequilibrate_with_matter(
    core: GenesisQuantumCore, matter: AdjointDiracMatterSector,
    y_bg: torch.Tensor, Omega_bg: torch.Tensor, c_matter: torch.Tensor,
    M_src: float, g: float, mu: float, nu: float, kappa: float, g1: float, g2: float,
    n_outer: int = 5, n_inner: int = 35
) -> Dict[str, object]:
    y, Omega = y_bg.clone(), Omega_bg.clone()
    Omega, Sx, _, _ = core.solve_hfb_omega(y, g=g, mu=mu, nu=nu, max_iter=15, Omega_init=Omega)

    def fast_tot_y(yv: torch.Tensor, Sx_t: torch.Tensor) -> float:
        e_y = core.mqm_y_energy_fixed_sx(yv, Sx_t, g=g, mu=mu, nu=nu)
        yr = yv.reshape(3, core.N2)
        Fm = (yr @ core.f_flat).reshape(3, core.N2, core.N2)
        Del = sum(Fm[I] @ Fm[I].T for I in range(3))
        lam = torch.clamp(safe_eigvalsh(0.5 * (Del + Del.T)), min=0.0)
        fv = g1 * lam + g2 * lam * lam
        return e_y + float(kappa * torch.sum(fv - torch.sqrt(fv * fv + 1.0))) + matter.matter_energy_only(yv, c_matter, M_src)

    if M_src > 0.0:
        dt = 0.008
        for _ in range(n_outer):
            prev_y, prev_g = None, None
            for _ in range(n_inner):
                grad_bg, _ = core.exact_grad_y(y, Sx, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2)
                _, grad_m = matter.matter_energy_and_grad(y, c_matter, M_src)
                grad = grad_bg + grad_m
                gn = float(torch.linalg.norm(grad))
                if gn < 1e-4:
                    break
                if prev_y is not None and prev_g is not None:
                    sv, dg = y - prev_y, grad - prev_g
                    sy, ss = float(torch.dot(sv, dg)), float(torch.dot(sv, sv))
                    if sy > 1e-12 and ss > 1e-16:
                        dt = float(np.clip(ss / sy, 1e-5, 0.15))
                E0 = fast_tot_y(y, Sx)
                step_sz = min(dt, 2.0 / max(gn, 1.0))
                accepted = False
                for _ in range(12):
                    y_try = core.project_traceless(y - step_sz * grad)
                    if fast_tot_y(y_try, Sx) <= E0 - 1e-4 * step_sz * (gn * gn):
                        prev_y, prev_g = y.clone(), grad.clone()
                        y = y_try
                        dt = min(step_sz * 1.35, 0.12)
                        accepted = True
                        break
                    step_sz *= 0.4
                if not accepted:
                    dt *= 0.25; prev_y, prev_g = None, None
            Omega, Sx, _, _ = core.solve_hfb_omega(y, g=g, mu=mu, nu=nu, max_iter=15, Omega_init=Omega)
            core.guard.check_runtime()

    grad_bg, E_tn = core.exact_grad_y(y, Sx, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2)
    E_m, grad_m = matter.matter_energy_and_grad(y, c_matter, M_src)
    e_mqm = core.mqm_quantum_energy(y, Omega, Sx, g=g, mu=mu, nu=nu)
    return {
        "M_src": float(M_src), "y": y, "Y": core.vec_to_mats(y), "Omega": Omega, "Sx": Sx,
        "E_mqm": e_mqm["E_mqm"], "E_tn": E_tn, "E_matter": E_m, "E_tot": e_mqm["E_mqm"] + E_tn + E_m,
        "grad_norm": float(torch.linalg.norm(grad_bg + grad_m)),
    }


def analyze_minimal_surface_gravity(
    core: GenesisQuantumCore, state: Dict[str, object],
    kappa: float, g1: float, g2: float, opt_steps: int = 100, n_blind_trials: int = 3, seed: int = 909
) -> Dict[str, object]:
    Y, Omega = state["Y"], state["Omega"]
    R_rms = float(torch.sqrt(torch.real(torch.sum(Y * Y.conj())) / float(core.N)))

    with torch.no_grad():
        _, _, f, E_mats, _ = core.bdg_link_sector(Y, kappa=kappa, Nf=1, g1=g1, g2=g2)
        cache = core.precompute_boson_entanglement_cache(Omega, Y=Y, project_gauss=True)
        ez, uz = safe_eigh(0.5 * (Y[2] + Y[2].conj().T))
        uz_desc = uz[:, torch.argsort(ez, descending=True)]
        uz_asc = uz[:, torch.argsort(ez, descending=False)]

    @torch.no_grad()
    def eval_subspace(Q_in: torch.Tensor) -> Dict[str, float]:
        M = Q_in.shape[1]
        Th = Q_in @ Q_in.conj().T
        cm = torch.stack([torch.real(torch.trace(Th @ Y[I])) / float(M) for I in range(3)])
        cm_norm = float(torch.linalg.norm(cm))
        Th_bar = core.eye_N - Th
        B_m = sum((Th @ Y[I] @ Th_bar) @ (Th_bar @ Y[I] @ Th) for I in range(3))
        ev_b = torch.clamp(torch.real(safe_eigvalsh(0.5 * (B_m + B_m.conj().T))), min=0.0)
        area_nc = float(np.sqrt(2.0) * torch.sum(torch.sqrt(ev_b)))
        S_tn = float(core.bdg_subregion_entropy(E_mats, f, Q_in, Nf=1))
        cas, S_bos = core.gauss_casimir_and_boson_entropy(Y, cache, Q_in, full_casimir=False)
        return {
            "S_ee": S_tn + float(S_bos), "S_tn": S_tn, "S_bos": float(S_bos), "casimir": float(cas),
            "area_nc": area_nc, "pol": cm_norm / max(R_rms, 1e-12),
            "cos_theta_axis": float(cm[2]) / max(cm_norm, 1e-12),
        }

    ns_compare = []
    for M in range(1, core.N // 2 + 1):
        rN, rS = eval_subspace(uz_desc[:, :M]), eval_subspace(uz_asc[:, :M])
        ns_compare.append({"M": M, "north": rN, "south": rS, "dS_N_minus_S": rN["S_ee"] - rS["S_ee"], "dArea_N_minus_S": rN["area_nc"] - rS["area_nc"]})

    gens = core.T_basis[1:].detach()
    blind_deflection = {}
    for M_test in [2, core.N // 2]:
        T_M = build_hermitian_basis(M_test, device=core.device)
        best_rec = None
        for tr in range(n_blind_trials):
            gen_q = torch.Generator(device="cpu").manual_seed(seed + 100 * M_test + 17 * tr)
            Z_q = (torch.randn((core.N, core.N), generator=gen_q, dtype=core.rdtype)
                   + 1j * torch.randn((core.N, core.N), generator=gen_q, dtype=core.rdtype)).to(core.device)
            U0, _ = torch.linalg.qr(Z_q)
            Q0 = U0[:, :M_test].detach()
            c_p = torch.zeros(gens.shape[0], dtype=core.rdtype, device=core.device, requires_grad=True)
            opt = torch.optim.Adam([c_p], lr=0.14)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps, eta_min=0.015)
            for _ in range(opt_steps):
                opt.zero_grad()
                H_g = torch.einsum("k,kij->ij", c_p.to(core.cdtype), gens)
                H_g = 0.5 * (H_g + H_g.conj().T)
                Q_in = torch.linalg.matrix_exp(1j * H_g) @ Q0
                loss = core.bdg_subregion_entropy(E_mats, f, Q_in, Nf=1) + core.gauss_casimir_and_boson_entropy(Y, cache, Q_in, T_M=T_M, full_casimir=False)[1]
                loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                H_g = torch.einsum("k,kij->ij", c_p.to(core.cdtype), gens)
                H_g = 0.5 * (H_g + H_g.conj().T)
                rec = eval_subspace(torch.linalg.matrix_exp(1j * H_g) @ Q0)
            if best_rec is None or rec["S_ee"] < best_rec["S_ee"]:
                best_rec = rec
        blind_deflection[f"M{M_test}"] = {"best": best_rec}

    return {"ns_compare": ns_compare, "blind_deflection": blind_deflection}


def plot_gravity_dashboard(N: int, sweep: List[Dict[str, object]], out_png: str):
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.8), dpi=140)
    fig.suptitle(f"Project Genesis-Rigorous (Unified Stage 3, N={N}): Emergent Gravity from Adjoint Dirac Matter", fontsize=14, fontweight="bold", y=0.98)

    rec0, recM = sweep[0], sweep[-1]
    th_deg = np.degrees(np.array(rec0["geom"]["thetas"]))
    r0, K0, g0 = np.array(rec0["geom"]["r_shell"]), np.array(rec0["geom"]["K_gauss"]), np.array(rec0["geom"]["sqrt_det_g"])
    Ms = np.array([r["M_src"] for r in sweep], dtype=float)
    cols = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(sweep) - 1, 1)))

    ax = axes[0, 0]
    for i, r in enumerate(sweep[1:]):
        ax.plot(th_deg, np.array(r["geom"]["r_shell"]) - r0, lw=2.2, color=cols[i], label=f"M_src={r['M_src']:.0f}")
    ax.axhline(0.0, color="k", ls=":"); ax.set_title("1. Radial Shift dr(theta)"); ax.set_xlabel("theta [deg]"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    th_r = np.array(rec0["geom"]["thetas"])
    th_f = np.concatenate([th_r, -th_r[::-1]])
    rM_ex = r0 + 5.0 * (np.array(recM["geom"]["r_shell"]) - r0)
    ax.plot(np.concatenate([r0, r0[::-1]]) * np.sin(th_f), np.concatenate([r0, r0[::-1]]) * np.cos(th_f), "k--", label="Vacuum")
    ax.plot(np.concatenate([rM_ex, rM_ex[::-1]]) * np.sin(th_f), np.concatenate([rM_ex, rM_ex[::-1]]) * np.cos(th_f), "b-", lw=2.2, label="Curved (5x)")
    ax.scatter([0.0], [rM_ex[0]], color="red", s=80, zorder=5, label="Dirac Matter")
    ax.set_aspect("equal"); ax.set_title("2. Meridian Cross-Section"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for i, r in enumerate(sweep[1:]):
        ax.plot(th_deg, 100.0 * (np.array(r["geom"]["sqrt_det_g"]) - g0) / np.maximum(g0, 1e-12), lw=2.2, color=cols[i], label=f"M_src={r['M_src']:.0f}")
    ax.axhline(0.0, color="k", ls=":"); ax.set_title("3. Provost-Vallee Metric Shift [%]"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i, r in enumerate(sweep[1:]):
        ax.plot(th_deg, 100.0 * (np.array(r["geom"]["K_gauss"]) - K0) / np.maximum(np.abs(K0), 1e-12), lw=2.2, color=cols[i], label=f"M_src={r['M_src']:.0f}")
    ax.axhline(0.0, color="k", ls=":"); ax.set_title("4. Gaussian Curvature Shift dK/K_0 [%]"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    KN = np.array([r["geom"]["K_north"] / rec0["geom"]["K_north"] for r in sweep])
    KS = np.array([r["geom"]["K_south"] / rec0["geom"]["K_south"] for r in sweep])
    ax.plot(Ms, KN, "r-o", lw=2.2, label="North Pole K_N / K_0 (Matter)")
    ax.plot(Ms, KS, "b--x", lw=1.8, label="South Pole K_S / K_0 (Control)")
    ax.set_title("5. Linear-Response Einstein Curvature Scaling"); ax.set_xlabel("M_src"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    dS = np.array([r["grav_ee"]["ns_compare"][1]["dS_N_minus_S"] for r in sweep])
    cos_m = np.array([r["grav_ee"]["blind_deflection"][f"M{N//2}"]["best"]["cos_theta_axis"] for r in sweep])
    ax2 = ax.twinx()
    l1 = ax.plot(Ms, dS, "m-o", lw=2.2, label="dS_EE(North-South, M=2)")
    l2 = ax2.plot(Ms, cos_m, "k-^", lw=2.0, label=f"Blind Hemisphere cos(theta) (M={N//2})")
    ax2.set_ylim(-1.15, 1.15); ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="best", fontsize=7.5)
    ax.set_title("6. Entanglement 1st Law & Blind Surface Deflection"); ax.set_xlabel("M_src"); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(out_png, dpi=150); plt.close(fig)


def run_phase2_gravity_experiment(
    state_pt: str = "", m_psi: float = 0.5, masses: List[float] = None,
    n_theta: int = 31, qrf_steps: int = 100, out_dir: str = "", prefer_cuda: bool = True
) -> Dict[str, object]:
    if not state_pt:
        cands = sorted(glob.glob(os.path.join("results", "*phase1_N8*", "phase1_states.pt")))
        if not cands:
            raise FileNotFoundError("No Phase 1 state file found. Run `genesis_cli.py phase1 --N 8` first.")
        state_pt = cands[-1]
    if masses is None:
        masses = [0.0, 10.0, 25.0, 50.0, 80.0]
    masses = sorted(set([0.0] + list(masses)))

    t0 = time.time()
    saved = torch.load(state_pt, weights_only=False)
    N = int(saved["N"])
    p = saved["params"]
    g, mu, nu, kappa, g1, g2 = float(p["g"]), float(p["mu"]), float(p["nu_emerg"]), float(p["kappa"]), float(p["g1"]), float(p["g2"])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join("results", f"unified_stage3_gravity_N{N}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    core = GenesisQuantumCore(N=N, D=3, guard=guard)
    matter = AdjointDiracMatterSector(core, m_psi=m_psi)

    y_bg = saved["universe_B"]["y"].to(core.device)
    Y_bg = saved["universe_B"]["Y"].to(core.device)
    Om_bg = saved["universe_B"]["Omega"].to(core.device)
    R_rms = float(torch.sqrt(torch.real(torch.sum(Y_bg * Y_bg.conj())) / float(N)))
    r_n, _ = find_shell_radius_along_ray(core, Y_bg, np.array([0.0, 0.0, 1.0]), r_hint=R_rms)
    _, c_north = matter.build_localized_probe_packet(Y_bg, torch.tensor([0.0, 0.0, r_n], dtype=core.rdtype, device=core.device))

    sweep = []
    for M_src in masses:
        st_m = reequilibrate_with_matter(core, matter, y_bg, Om_bg, c_north, M_src, g, mu, nu, kappa, g1, g2)
        geom_m = scan_meridian_geometry(core, st_m["Y"], r_hint=r_n, n_theta=n_theta)
        grav_ee = analyze_minimal_surface_gravity(core, st_m, kappa, g1, g2, opt_steps=qrf_steps)
        print(
            f"[STAGE 3 M={M_src:4.1f}] K_N/K_S={geom_m['K_north']/max(geom_m['K_south'],1e-18):.4f} | "
            f"PV_N/S={geom_m['g_area_north']/max(geom_m['g_area_south'],1e-18):.4f} | "
            f"dS(N-S,M=2)={grav_ee['ns_compare'][1]['dS_N_minus_S']:+.4f} | "
            f"BlindM{N//2} cos={grav_ee['blind_deflection'][f'M{N//2}']['best']['cos_theta_axis']:+.3f}"
        )
        sweep.append({
            "M_src": M_src, "E_tot": st_m["E_tot"], "E_matter": st_m["E_matter"],
            "grad_norm": st_m["grad_norm"], "geom": geom_m, "grav_ee": grav_ee,
        })

    M_arr = np.array([r["M_src"] for r in sweep], dtype=float)
    r0 = sweep[0]["geom"]
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0

    lin = {
        "corr_M_vs_dK_north": corr(M_arr, np.array([r["geom"]["K_north"] - r0["K_north"] for r in sweep])),
        "corr_M_vs_dg_area_north": corr(M_arr, np.array([r["geom"]["g_area_north"] - r0["g_area_north"] for r in sweep])),
        "corr_M_vs_dS_NS_M2": corr(M_arr, np.array([r["grav_ee"]["ns_compare"][1]["dS_N_minus_S"] for r in sweep])),
    }
    plot_gravity_dashboard(N, sweep, os.path.join(out_dir, "stage3_gravity_dashboard.png"))
    ram_mb, vram_mb = guard.check_runtime()
    report = {"N": N, "timestamp": ts, "out_dir": out_dir, "state_pt": state_pt, "linear_response": lin, "sweep": sweep, "stats": {"elapsed_sec": time.time() - t0, "ram_mb": ram_mb, "vram_mb": vram_mb}}
    with open(os.path.join(out_dir, "stage3_gravity_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    guard.cleanup()
    return report
