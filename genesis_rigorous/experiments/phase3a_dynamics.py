"""
Phase 3A Experiment: Two-Body Curvature-Mediated Gravitational Potential V_grav(theta_12)
+ Real-Time Symplectic TD-HFB Causal Light-Cone & Curvature Wave Propagation.
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

from ..dirac_matter import AdjointDiracMatterSector, find_shell_radius_along_ray
from ..exact_hfb import GenesisQuantumCore
from ..safety import HardwareSafetyGuard, SafetyConfig, safe_eigvalsh


@torch.no_grad()
def run_phase3a_dynamics_experiment(
    state_pt: str = "", m_psi: float = 0.5, M_body: float = 35.0, M_pulse: float = 120.0,
    dt: float = 0.004, n_steps: int = 420, out_dir: str = "", prefer_cuda: bool = True
) -> Dict[str, object]:
    if not state_pt:
        cands = sorted(glob.glob(os.path.join("results", "*phase1_N8*", "phase1_states.pt")))
        if not cands:
            raise FileNotFoundError("No Phase 1 state file found. Run `genesis_cli.py phase1 --N 8` first.")
        state_pt = cands[-1]

    t0 = time.time()
    saved = torch.load(state_pt, weights_only=False)
    N = int(saved["N"])
    p = saved["params"]
    g, mu, nu, kappa, g1, g2 = float(p["g"]), float(p["mu"]), float(p["nu_emerg"]), float(p["kappa"]), float(p["g1"]), float(p["g2"])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join("results", f"unified_phase3a_dynamics_N{N}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    core = GenesisQuantumCore(N=N, D=3, guard=guard)
    matter = AdjointDiracMatterSector(core, m_psi=m_psi)

    y_bg = saved["universe_B"]["y"].to(core.device)
    Y_bg = saved["universe_B"]["Y"].to(core.device)
    Om_bg = saved["universe_B"]["Omega"].to(core.device)
    R_rms = float(torch.sqrt(torch.real(torch.sum(Y_bg * Y_bg.conj())) / float(N)))
    r_shell, _ = find_shell_radius_along_ray(core, Y_bg, np.array([0.0, 0.0, 1.0]), r_hint=R_rms)
    alpha_bg = r_shell / (0.5 * (N - 1))

    # --- 1. Two-Body Emergent Gravitational Potential ---
    _, c1 = matter.build_localized_probe_packet(Y_bg, torch.tensor([0.0, 0.0, r_shell], dtype=core.rdtype, device=core.device))
    e1_bg = matter.matter_energy_only(y_bg, c1, M_body)
    Om_ref, Sx_ref, _, _ = core.solve_hfb_omega(y_bg, g=g, mu=mu, nu=nu, max_iter=15, Omega_init=Om_bg)
    E_bg = core.mqm_quantum_energy(y_bg, Om_ref, Sx_ref, g=g, mu=mu, nu=nu)["E_mqm"] + float(core.bdg_link_sector(Y_bg, kappa=kappa, g1=g1, g2=g2)[0])

    def fast_2b_y(yv, Sx_t, c2_v):
        e_y = core.mqm_y_energy_fixed_sx(yv, Sx_t, g=g, mu=mu, nu=nu)
        yr = yv.reshape(3, core.N2)
        Fm = (yr @ core.f_flat).reshape(3, core.N2, core.N2)
        Del = sum(Fm[I] @ Fm[I].T for I in range(3))
        lam = torch.clamp(safe_eigvalsh(0.5 * (Del + Del.T)), min=0.0)
        fv = g1 * lam + g2 * lam * lam
        return e_y + float(kappa * torch.sum(fv - torch.sqrt(fv * fv + 1.0))) + matter.matter_energy_only(yv, c1, M_body) + matter.matter_energy_only(yv, c2_v, M_body)

    angles_deg = [25.0, 45.0, 65.0, 90.0, 130.0, 180.0]
    recs_2b = []
    for ang in angles_deg:
        th = np.radians(ang)
        _, c2 = matter.build_localized_probe_packet(Y_bg, torch.tensor([r_shell * np.sin(th), 0.0, r_shell * np.cos(th)], dtype=core.rdtype, device=core.device))
        E_froz = E_bg + e1_bg + matter.matter_energy_only(y_bg, c2, M_body)
        y, Omega, Sx = y_bg.clone(), Om_ref.clone(), Sx_ref.clone()
        step_dt = 0.008
        for _ in range(5):
            prev_y, prev_g = None, None
            for _ in range(35):
                g_bg, _ = core.exact_grad_y(y, Sx, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2)
                _, g1m = matter.matter_energy_and_grad(y, c1, M_body)
                _, g2m = matter.matter_energy_and_grad(y, c2, M_body)
                grad = g_bg + g1m + g2m
                gn = float(torch.linalg.norm(grad))
                if gn < 1e-4: break
                if prev_y is not None and prev_g is not None:
                    sv, dg = y - prev_y, grad - prev_g
                    sy, ss = float(torch.dot(sv, dg)), float(torch.dot(sv, sv))
                    if sy > 1e-12 and ss > 1e-16: step_dt = float(np.clip(ss / sy, 1e-5, 0.15))
                E0 = fast_2b_y(y, Sx, c2)
                ssz = min(step_dt, 2.0 / max(gn, 1.0))
                acc = False
                for _ in range(10):
                    yt = core.project_traceless(y - ssz * grad)
                    if fast_2b_y(yt, Sx, c2) <= E0 - 1e-4 * ssz * gn * gn:
                        prev_y, prev_g, y = y.clone(), grad.clone(), yt
                        step_dt = min(ssz * 1.35, 0.12); acc = True; break
                    ssz *= 0.4
                if not acc: step_dt *= 0.25; prev_y, prev_g = None, None
            Omega, Sx, _, _ = core.solve_hfb_omega(y, g=g, mu=mu, nu=nu, max_iter=15, Omega_init=Omega)
        E_tot = (
            core.mqm_quantum_energy(y, Omega, Sx, g=g, mu=mu, nu=nu)["E_mqm"]
            + float(core.bdg_link_sector(core.vec_to_mats(y), kappa=kappa, g1=g1, g2=g2)[0])
            + matter.matter_energy_only(y, c1, M_body) + matter.matter_energy_only(y, c2, M_body)
        )
        recs_2b.append({"theta_deg": ang, "geodesic_dist": float(r_shell * th), "E_frozen": E_froz, "E_tot": E_tot})

    E_ref, Ef_ref = recs_2b[-1]["E_tot"], recs_2b[-1]["E_frozen"]
    for r in recs_2b:
        r["V_grav"] = r["E_tot"] - E_ref
        r["V_frozen_control"] = r["E_frozen"] - Ef_ref
        print(f"  [3A-1 theta={r['theta_deg']:5.1f}°] Frozen={r['V_frozen_control']:+.2e} | V_grav={r['V_grav']:+.4f}")

    # --- 2. Real-Time Symplectic TD-HFB Causal Light Cone ---
    thetas = np.linspace(0.0, np.pi, 19)
    rhos = []
    for th in thetas:
        rho_th, _ = matter.build_localized_probe_packet(Y_bg, torch.tensor([r_shell * np.sin(th), 0.0, r_shell * np.cos(th)], dtype=core.rdtype, device=core.device))
        rhos.append(rho_th / torch.clamp(torch.real(torch.trace(rho_th)), min=1e-12))
    rhos_stack = torch.stack(rhos, dim=0)

    t_pulse, tau_pulse = 0.06, 0.022
    def force_and_pot(yv, t_val):
        g_bg, E_tn = core.exact_grad_y(yv, Sx_ref, g=g, mu=mu, nu=nu, kappa=kappa, g1=g1, g2=g2)
        z = (t_val - t_pulse) / tau_pulse
        m_t = float(M_pulse * np.exp(-0.5 * z * z)) if abs(z) <= 8.0 else 0.0
        E_m, g_m = matter.matter_energy_and_grad(yv, c1, m_t) if m_t > 0 else (0.0, torch.zeros_like(yv))
        return -(g_bg + g_m), core.mqm_y_energy_fixed_sx(yv, Sx_ref, g=g, mu=mu, nu=nu) + E_tn + E_m

    y, q = y_bg.clone(), torch.zeros_like(y_bg)
    force, E_pot = force_and_pot(y, 0.0)
    times, etots, q2_grid = [], [], []
    for step in range(n_steps + 1):
        t_cur = step * dt
        if step % 3 == 0:
            Q_cur = core.vec_to_mats(q)
            times.append(t_cur)
            etots.append(0.5 * float(torch.dot(q, q)) + E_pot)
            Q2 = sum(Q_cur[I] @ Q_cur[I] for I in range(3))
            q2_grid.append(torch.real(torch.einsum("tij,ji->t", rhos_stack, Q2)).cpu().numpy().tolist())
        if step == n_steps: break
        q_h = q + 0.5 * dt * force
        y = core.project_traceless(y + dt * q_h)
        force, E_pot = force_and_pot(y, t_cur + dt)
        q = core.project_traceless(q_h + 0.5 * dt * force)

    ts_arr, et_arr, q2_arr = np.array(times), np.array(etots), np.array(q2_grid)
    E_free = et_arr[ts_arr >= 0.24]
    rel_cons = float(np.max(np.abs(E_free - np.mean(E_free))) / max(abs(float(np.mean(E_free) - et_arr[0])), 1e-12))

    win = (ts_arr >= 0.04) & (ts_arr <= 1.35)
    t_win, q2_w = ts_arr[win], q2_arr[win]
    t_onset = [float(t_win[int(np.argmax(q2_w[:, j] >= 0.15 * max(float(np.max(q2_w[:, j])), 1e-12)))]) for j in range(len(thetas))]
    fit_m = (thetas >= np.radians(25.0)) & (thetas <= np.radians(165.0))
    slope, _ = np.polyfit(thetas[fit_m], np.array(t_onset)[fit_m], 1)
    c_ang = float(1.0 / max(slope, 1e-12))
    r_lc = float(np.corrcoef(thetas[fit_m], np.array(t_onset)[fit_m])[0, 1])
    c_th = g * alpha_bg
    print(f"  [3A-2 LIGHT CONE] Energy drift={rel_cons:.2e} | c_ang={c_ang:.3f} rad/t (r={r_lc:.4f}) vs g*alpha={c_th:.3f} (ratio={c_ang/c_th:.3f})")

    # Save 3-panel summary figure
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=140)
    fig.suptitle(f"Project Genesis-Rigorous (Unified Phase 3A, N={N}): 2-Body Gravity & Real-Time Light Cone", fontsize=13.5, fontweight="bold", y=0.98)
    axes[0].plot([r["theta_deg"] for r in recs_2b], [r["V_grav"] for r in recs_2b], "b-o", lw=2.3, label="Curved V_grav(theta)")
    axes[0].plot([r["theta_deg"] for r in recs_2b], [r["V_frozen_control"] for r in recs_2b], "r--x", label="Frozen Control")
    axes[0].axhline(0, color="k", ls=":"); axes[0].set_title("1. Two-Body Emergent Gravity"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    q2_n = q2_arr / max(float(np.max(q2_arr)), 1e-12)
    im = axes[1].imshow(np.log10(np.maximum(q2_n, 1e-5)).T, origin="lower", aspect="auto", extent=[ts_arr[0], ts_arr[-1], 0, 180], cmap="magma", vmin=-4, vmax=0)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("2. Real-Time Emergent Light Cone"); axes[1].set_xlabel("Time t"); axes[1].set_ylabel("theta [deg]")

    axes[2].plot(np.degrees(thetas), t_onset, "ro-", label=f"Onset c={c_ang:.2f} rad/t (r={r_lc:.3f})")
    axes[2].plot(np.degrees(thetas), 0.06 + thetas / c_th, "k:", lw=2.0, label=f"YM Theory g*alpha={c_th:.2f}")
    axes[2].set_title("3. Emergent Speed of Light Fit"); axes[2].set_xlabel("theta [deg]"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(os.path.join(out_dir, "phase3a_dynamics_dashboard.png"), dpi=150); plt.close(fig)

    ram_mb, vram_mb = guard.check_runtime()
    report = {
        "N": N, "timestamp": ts, "out_dir": out_dir,
        "two_body": {"V_grav_near": recs_2b[0]["V_grav"], "max_frozen_err": max(abs(r["V_frozen_control"]) for r in recs_2b), "records": recs_2b},
        "light_cone": {"rel_energy_conservation": rel_cons, "c_ang_onset": c_ang, "c_ang_theory": c_th, "r_corr": r_lc},
        "stats": {"elapsed_sec": time.time() - t0, "ram_mb": ram_mb, "vram_mb": vram_mb},
    }
    with open(os.path.join(out_dir, "phase3a_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    guard.cleanup()
    return report
