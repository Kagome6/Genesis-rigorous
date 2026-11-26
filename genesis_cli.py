#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROJECT GENESIS-RIGOROUS v1.1.0 — UNIFIED CLI ENTRYPOINT (genesis_cli.py)
================================================================================
Zero-Cheat Matrix Quantum Mechanics x Quantum Entanglement Laboratory for
Emergent Spacetime, Gravity, Non-Abelian Gauge Bosons, Chiral Fermions,
Random-Law Ecosystems, and Non-Perturbative Matrix Bootstrap.

Subcommands:
  verify         Run all 20 exact mathematical, gauge-invariance & bootstrap audits (<6s)
  phase1         Dual-Universe phase transition + Point-Probe + Blind QRF Area Law
  gravity        Stage 3 localized Adjoint Dirac matter curvature & surface deflection
  realtime       Phase 3A two-body emergent gravity attraction & real-time light cone
  large-n        Phase 3B O(N^3) matrix-native continuum scaling (N=16, 20, 24, 32)
  gauge          Phase 3C emergent U(2) gauge bosons & Coulomb-branch Matrix Higgs
  chiral         Phase 3D 4D S^2xS^2 emergence & fuzzy Ginsparg-Wilson chiral generations
  watch          Interactive zero-cheat live GUI observatory (emergence | metastable | wave)
  law-ecosystem  Level-2A: 200 random bounded 3-matrix laws across 4 controlled ensembles
  dirty-healing  Level-2B: Dirty-law survival & large-N RG robustness (N=6, 8, 12, 16)
  bootstrap      Level-2C: Han-Hartnoll-Kruthoff (2020) Matrix & Yang-Mills SDP Bootstrap
  suite          Run fast end-to-end showcase across verify + phase1 + gauge + chiral
================================================================================
"""

import argparse
import datetime
import json
import os
import sys
import time

from genesis_rigorous import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genesis_cli.py",
        description=f"Project Genesis-Rigorous v{__version__} — Unified Emergent Spacetime & Law Discovery Lab",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution instead of CUDA")
    sub = parser.add_subparsers(dest="command", required=True)

    # verify (20-test master audit)
    p_ver = sub.add_parser("verify", help="Run master 20-test mathematical, gauge-invariance & bootstrap audit")
    p_ver.add_argument("--save-cert", type=str, default="", help="Optional path to save JSON audit certificate")

    # phase1
    p_p1 = sub.add_parser("phase1", help="Phase 1: Dual-Universe transition + Point-Probe + Blind QRF Area Law")
    p_p1.add_argument("--N", type=int, default=8)
    p_p1.add_argument("--steps", type=int, default=240)
    p_p1.add_argument("--restarts", type=int, default=4)
    p_p1.add_argument("--restarts-chaos", type=int, default=2)
    p_p1.add_argument("--qrf-steps", type=int, default=100)
    p_p1.add_argument("--out-dir", type=str, default="")

    # gravity (Stage 3)
    p_gr = sub.add_parser("gravity", help="Phase 2 / Stage 3: Single-body Adjoint Dirac matter backreaction")
    p_gr.add_argument("--state-pt", type=str, default="")
    p_gr.add_argument("--masses", type=str, default="0,10,25,50,80")
    p_gr.add_argument("--m-psi", type=float, default=0.5)
    p_gr.add_argument("--n-theta", type=int, default=31)
    p_gr.add_argument("--qrf-steps", type=int, default=100)
    p_gr.add_argument("--out-dir", type=str, default="")

    # realtime (Phase 3A)
    p_rt = sub.add_parser("realtime", help="Phase 3A: Two-body emergent gravity & real-time causal light cone")
    p_rt.add_argument("--state-pt", type=str, default="")
    p_rt.add_argument("--M-body", type=float, default=35.0)
    p_rt.add_argument("--M-pulse", type=float, default=120.0)
    p_rt.add_argument("--n-steps", type=int, default=450)
    p_rt.add_argument("--out-dir", type=str, default="")

    # large-n (Phase 3B)
    p_ln = sub.add_parser("large-n", help="Phase 3B: O(N^3) Matrix-Native Large-N continuum scaling (N=16..32)")
    p_ln.add_argument("--Ns", type=str, default="16,20,24,32")
    p_ln.add_argument("--out-dir", type=str, default="")

    # gauge (Phase 3C)
    p_ga = sub.add_parser("gauge", help="Phase 3C: Emergent U(2) gauge bosons & Coulomb-branch Matrix Higgs")
    p_ga.add_argument("--n-sphere", type=int, default=6)
    p_ga.add_argument("--out-dir", type=str, default="")

    # chiral (Phase 3D)
    p_ch = sub.add_parser("chiral", help="Phase 3D: 4D S^2xS^2 emergence & fuzzy Ginsparg-Wilson chiral generations")
    p_ch.add_argument("--out-dir", type=str, default="")

    # watch (Interactive Zero-Cheat Live Observatory)
    p_wa = sub.add_parser("watch", help="Launch interactive zero-cheat GUI observatory (emergence | metastable | wave)")
    p_wa.add_argument("--mode", type=str, default="emergence", choices=["emergence", "metastable", "wave"])
    p_wa.add_argument("--N", type=int, default=8)
    p_wa.add_argument("--frames", type=int, default=68)
    p_wa.add_argument("--fps", type=int, default=16)
    p_wa.add_argument("--state-pt", type=str, default="")
    p_wa.add_argument("--save-gif", type=str, default="")
    p_wa.add_argument("--headless", action="store_true")

    # law-ecosystem (Level-2A)
    p_le = sub.add_parser("law-ecosystem", help="Level-2A: 200 random bounded 3-matrix laws ecosystem scan")
    p_le.add_argument("--N", type=int, default=8)
    p_le.add_argument("--n-per-ensemble", type=int, default=50)
    p_le.add_argument("--out-dir", type=str, default="")

    # dirty-healing (Level-2B)
    p_dh = sub.add_parser("dirty-healing", help="Level-2B: Dirty-law survival & large-N RG robustness (N=6..16)")
    p_dh.add_argument("--Ns", type=str, default="6,8,12,16")
    p_dh.add_argument("--out-dir", type=str, default="")

    # bootstrap (Level-2C)
    p_bs = sub.add_parser("bootstrap", help="Level-2C: Han-Hartnoll-Kruthoff (2020) Matrix & Yang-Mills SDP Bootstrap")
    p_bs.add_argument("--out-dir", type=str, default="")

    # suite (End-to-end showcase)
    p_su = sub.add_parser("suite", help="Run integrated end-to-end showcase (verify + phase1 + gauge + chiral)")
    p_su.add_argument("--fast", action="store_true", help="Fast smoke-test mode (N=6 phase1 + gauge + chiral, ~90s)")
    p_su.add_argument("--out-dir", type=str, default="")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    prefer_cuda = not args.cpu

    # Lazy-import interactive GUI viewer BEFORE any module sets matplotlib Agg backend
    if args.command == "watch":
        from genesis_rigorous.experiments.spacetime_viewer import run_spacetime_viewer
        run_spacetime_viewer(
            mode=args.mode, N=args.N, n_frames=args.frames, state_pt=args.state_pt,
            fps=args.fps, save_gif=args.save_gif, headless=args.headless, prefer_cuda=prefer_cuda,
        )
        return

    from genesis_rigorous.experiments import (
        run_phase1_experiment,
        run_phase2_gravity_experiment,
        run_phase3a_dynamics_experiment,
        run_phase3b_large_n_experiment,
        run_phase3c_gauge_experiment,
        run_phase3d_chiral_experiment,
        run_law_ecosystem_experiment,
        run_dirty_law_healing_experiment,
        run_matrix_bootstrap_experiment,
    )
    from genesis_rigorous.verification import run_master_verification

    if args.command == "verify":
        res = run_master_verification(prefer_cuda=prefer_cuda)
        if args.save_cert:
            os.makedirs(os.path.dirname(args.save_cert) or ".", exist_ok=True)
            with open(args.save_cert, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"[SAVED] Audit certificate: {args.save_cert}")
        sys.exit(0 if res["all_passed"] else 1)

    elif args.command == "phase1":
        run_phase1_experiment(
            N=args.N, steps=args.steps, restarts=args.restarts, restarts_chaos=args.restarts_chaos,
            qrf_steps=args.qrf_steps, out_dir=args.out_dir, prefer_cuda=prefer_cuda,
        )

    elif args.command == "gravity":
        masses = [float(x.strip()) for x in args.masses.split(",") if x.strip()]
        run_phase2_gravity_experiment(
            state_pt=args.state_pt, m_psi=args.m_psi, masses=masses,
            n_theta=args.n_theta, qrf_steps=args.qrf_steps, out_dir=args.out_dir, prefer_cuda=prefer_cuda,
        )

    elif args.command == "realtime":
        run_phase3a_dynamics_experiment(
            state_pt=args.state_pt, M_body=args.M_body, M_pulse=args.M_pulse,
            n_steps=args.n_steps, out_dir=args.out_dir, prefer_cuda=prefer_cuda,
        )

    elif args.command == "large-n":
        Ns = [int(x.strip()) for x in args.Ns.split(",") if x.strip()]
        run_phase3b_large_n_experiment(N_list=Ns, out_dir=args.out_dir, prefer_cuda=prefer_cuda)

    elif args.command == "gauge":
        run_phase3c_gauge_experiment(n_sphere=args.n_sphere, out_dir=args.out_dir, prefer_cuda=prefer_cuda)

    elif args.command == "chiral":
        run_phase3d_chiral_experiment(out_dir=args.out_dir, prefer_cuda=prefer_cuda)

    elif args.command == "law-ecosystem":
        run_law_ecosystem_experiment(
            N=args.N, n_per_ensemble=args.n_per_ensemble, out_dir=args.out_dir, prefer_cuda=prefer_cuda,
        )

    elif args.command == "dirty-healing":
        Ns = [int(x.strip()) for x in args.Ns.split(",") if x.strip()]
        run_dirty_law_healing_experiment(Ns=Ns, out_dir=args.out_dir, prefer_cuda=prefer_cuda)

    elif args.command == "bootstrap":
        run_matrix_bootstrap_experiment(out_dir=args.out_dir)

    elif args.command == "suite":
        t0 = time.time()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = args.out_dir or os.path.join("results", f"unified_suite_{ts}")
        os.makedirs(root, exist_ok=True)
        print("=" * 92)
        print(f"PROJECT GENESIS-RIGOROUS v{__version__} — INTEGRATED SHOWCASE SUITE (fast={args.fast})")
        print(f"Archive Root: {root}")
        print("=" * 92)

        ver = run_master_verification(prefer_cuda=prefer_cuda)
        if not ver["all_passed"]:
            print("[ABORT] Master verification failed!")
            sys.exit(1)

        N_p1 = 6 if args.fast else 8
        p1 = run_phase1_experiment(
            N=N_p1, steps=(180 if args.fast else 240), restarts=(3 if args.fast else 4),
            restarts_chaos=2, qrf_steps=(70 if args.fast else 100),
            out_dir=os.path.join(root, f"phase1_N{N_p1}"), prefer_cuda=prefer_cuda,
        )
        p3c = run_phase3c_gauge_experiment(
            n_sphere=6, out_dir=os.path.join(root, "phase3c_gauge"), prefer_cuda=prefer_cuda
        )
        p3d = run_phase3d_chiral_experiment(
            out_dir=os.path.join(root, "phase3d_chiral"), prefer_cuda=prefer_cuda
        )
        p2 = None
        if not args.fast:
            p2 = run_phase2_gravity_experiment(
                state_pt=p1["state_pt"], masses=[0.0, 25.0, 60.0], n_theta=25, qrf_steps=90,
                out_dir=os.path.join(root, "stage3_gravity"), prefer_cuda=prefer_cuda,
            )

        summary = {
            "version": __version__,
            "timestamp": ts,
            "fast_mode": args.fast,
            "elapsed_sec": time.time() - t0,
            "verify": {"passed": ver["passed_count"], "total": ver["total_count"], "all_passed": ver["all_passed"]},
            "phase1": {"N": N_p1, "E_B": p1["universe_B"]["E_tot"], "tr_A": p1["universe_A"]["diag"]["tr_Y2"], "alg_err_B": p1["universe_B"]["diag"]["rel_alg_err"], "area_law_B": p1["universe_B"]["area_law_stats"]},
            "phase3c_gauge": {"max_higgs_err": p3c["max_higgs_err"], "max_photon_drift": p3c["max_photon_drift"]},
            "phase3d_chiral": {"emergent_dim": p3d["emergent_4d"]["emergent_dim"], "metric_6x6": p3d["emergent_4d"]["metric_6x6_evals"], "q3_noisy_index": [r for r in p3d["chiral_fermions"] if r["q"] == 3 and r["noise"] > 0][0]["chiral_index"]},
        }
        if p2 is not None:
            summary["stage3_gravity"] = p2["linear_response"]
        out_sum = os.path.join(root, "suite_summary.json")
        with open(out_sum, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("=" * 92)
        print(f"[SUITE COMPLETE] Elapsed: {time.time()-t0:.1f}s | Master Summary: {out_sum}")
        print("=" * 92)


if __name__ == "__main__":
    main()
