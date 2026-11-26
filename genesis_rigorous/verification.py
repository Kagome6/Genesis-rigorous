"""
Master 17-Test Mathematical & Gauge-Invariance Verification Suite for Project Genesis-Rigorous.
Verifies every physics engine (Exact HFB, BdG, Gauss Casimir, Cholesky S_bos, Point-Probe,
Adjoint Dirac + Daleckii-Krein, O(N^3) Matrix-Native, 4D Gauge/Higgs, 6D Clifford, GKP Chiral Index)
to machine precision / statistical exactness in < 5 seconds on GPU.
"""

import time
from typing import Dict

import numpy as np
import torch

from .dirac_matter import AdjointDiracMatterSector
from .exact_hfb import GenesisQuantumCore, build_hermitian_basis, build_su2_irrep
from .experiments.phase3c_gauge import build_exact_4d_hessian_matrix_free
from .experiments.phase3d_chiral import build_6d_clifford_gammas, build_exact_gkp_chiral_dirac
from .matrix_native import MatrixNativeQuantumCore
from .safety import HardwareSafetyGuard, SafetyConfig


def _rand_haar(N: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    Z = (torch.randn((N, N), generator=gen, dtype=torch.float64) + 1j * torch.randn((N, N), generator=gen, dtype=torch.float64)).to(device)
    U, R = torch.linalg.qr(Z)
    return U * (torch.diag(R) / torch.abs(torch.diag(R)))


def run_master_verification(prefer_cuda: bool = True) -> Dict[str, object]:
    t0 = time.time()
    guard = HardwareSafetyGuard(SafetyConfig(), prefer_cuda=prefer_cuda)
    device = guard.device
    results = {}

    print("=" * 92)
    print(f"PROJECT GENESIS-RIGOROUS v1.1.0 — MASTER 20-TEST MATHEMATICAL AUDIT (Device={device})")
    print("=" * 92)

    def record(tid: str, desc: str, passed: bool, detail: str):
        results[tid] = {"description": desc, "passed": bool(passed), "detail": detail}
        print(f"[{tid}] {desc:42s}: {detail} ({'PASS' if passed else 'FAIL'})")

    # T01: Hardware Safety Preflight Blocks
    ok_b1, ok_b2 = False, False
    try:
        guard.preflight_check(N=64, D=3, mode="exact_hfb")
    except MemoryError:
        ok_b1 = True
    try:
        guard.preflight_check(N=100, D=3, mode="matrix_native")
    except MemoryError:
        ok_b2 = True
    record("T01", "Preflight Safety Block (N=64 / N=100)", ok_b1 and ok_b2, f"exact_hfb={ok_b1}, native={ok_b2}")

    # Core at N=4 for fast exact tests
    N = 4
    core = GenesisQuantumCore(N=N, D=3, guard=guard)
    gen = torch.Generator(device="cpu").manual_seed(2026)
    g_t, mu_t, nu_t = 0.4, 1.1, 1.7

    # T02: Isotropic 4th-order Wick E_YM^(4)
    w0 = 1.3
    sig2 = 0.5 / w0
    y_zero = torch.zeros(core.nB, dtype=core.rdtype, device=device)
    Om_iso = w0 * core.eye_nB
    Sx_iso = sig2 * core.eye_nB
    e_iso = core.mqm_quantum_energy(y_zero, Om_iso, Sx_iso, g=g_t, mu=mu_t, nu=nu_t)["E_ym4"]
    an_ym4 = 0.5 * (g_t ** 2) * 6 * N * (N * N - 1) * (sig2 ** 2)
    err_t02 = abs(e_iso - an_ym4)
    record("T02", "Isotropic 4th-Order Wick E_YM^(4)", err_t02 < 1e-12, f"Err={err_t02:.2e}")

    # T03: Anisotropic Sigma_x GPU Monte Carlo (200,000 samples)
    core3 = GenesisQuantumCore(N=3, D=3, guard=guard)
    gen_mc = torch.Generator(device="cpu").manual_seed(202602)
    A3 = torch.randn((core3.nB, core3.nB), generator=gen_mc, dtype=core3.rdtype).to(device) * 0.25
    Om3 = A3 @ A3.T + 1.2 * core3.eye_nB
    ew3, ev3 = torch.linalg.eigh(0.5 * (Om3 + Om3.T))
    Sx3 = (ev3 * (0.5 / ew3)) @ ev3.T
    L_c = torch.linalg.cholesky(0.5 * (Sx3 + Sx3.T))
    e_wick3 = core3.mqm_quantum_energy(torch.zeros(core3.nB, dtype=core3.rdtype, device=device), Om3, Sx3, g=g_t, mu=mu_t, nu=nu_t)["E_ym4"]
    n_mc, bs = 200000, 20000
    s1, s2 = 0.0, 0.0
    for _ in range(n_mc // bs):
        zs = torch.randn((bs, core3.nB), generator=gen_mc, dtype=core3.rdtype).to(device)
        dx = (zs @ L_c.T).reshape(bs, 3, core3.N2)
        F_mc = torch.matmul(dx, core3.f_flat).reshape(bs, 3, core3.N2, core3.N2)
        C_mc = torch.einsum("nibc,njb->nijc", F_mc, dx)
        vals = 0.25 * (g_t ** 2) * torch.sum(C_mc * C_mc, dim=(1, 2, 3))
        s1 += float(torch.sum(vals)); s2 += float(torch.sum(vals * vals))
    mc_m = s1 / n_mc
    mc_se = float(np.sqrt(max(s2 / n_mc - mc_m * mc_m, 0.0) / n_mc))
    z_sc = abs(mc_m - e_wick3) / max(mc_se, 1e-15)
    record("T03", "Anisotropic Wick vs GPU MC (200k)", z_sc < 3.5, f"An={e_wick3:.5f}, MC={mc_m:.5f}+/-{mc_se:.5f} ({z_sc:.2f}σ)")

    # T04: Free Vacuum (Y=0) Casimir & S_bos == 0
    Q_0 = torch.eye(N, dtype=core.cdtype, device=device)[:, :2]
    c_iso = core.precompute_boson_entanglement_cache(Om_iso, Y=core.vec_to_mats(y_zero), project_gauss=True)
    cas_v, sb_v = core.gauss_casimir_and_boson_entropy(core.vec_to_mats(y_zero), c_iso, Q_0, full_casimir=True)
    record("T04", "Free Vacuum (Y=0) Casimir & S_bos == 0", float(cas_v) < 1e-13 and float(sb_v) < 1e-13, f"Cas={float(cas_v):.2e}, S_bos={float(sb_v):.2e}")

    # T05 & T06: Curved Full-Universe (M=N) Gauss Zero & Full U(N) Gauge Invariance (7 obs)
    y_r = torch.randn(core.nB, generator=gen, dtype=core.rdtype).to(device) * 0.6
    Y_r = core.vec_to_mats(y_r)
    A_om = torch.randn((core.nB, core.nB), generator=gen, dtype=core.rdtype).to(device) * 0.25
    Om_r = A_om @ A_om.T + 1.5 * core.eye_nB
    ew_r, ev_r = torch.linalg.eigh(0.5 * (Om_r + Om_r.T))
    Sx_r = (ev_r * (0.5 / ew_r)) @ ev_r.T
    U_h = _rand_haar(N, gen, device)
    T_rot = torch.einsum("ij,bjk,lk->bil", U_h, core.T_basis, U_h.conj())
    R_adj = torch.real(torch.einsum("aji,bij->ab", core.T_basis, T_rot)).contiguous()
    R_full = torch.kron(torch.eye(3, dtype=core.rdtype, device=device), R_adj)
    Y_rot = torch.einsum("ij,djk,lk->dil", U_h, Y_r, U_h.conj()).contiguous()
    y_rot = core.mats_to_vec(Y_rot)
    Om_rot, Sx_rot, Q_rot = R_full @ Om_r @ R_full.T, R_full @ Sx_r @ R_full.T, U_h @ Q_0

    c1 = core.precompute_boson_entanglement_cache(Om_r, Y=Y_r, project_gauss=True)
    c2 = core.precompute_boson_entanglement_cache(Om_rot, Y=Y_rot, project_gauss=True)
    cas_MN, _ = core.gauss_casimir_and_boson_entropy(Y_r, c1, torch.eye(N, dtype=core.cdtype, device=device), full_casimir=False)
    record("T05", "Curved Full-Universe (M=N) Gauss Zero", float(cas_MN) < 1e-11, f"Cas_Term1(M=N)={float(cas_MN):.2e}")

    e1 = core.mqm_quantum_energy(y_r, Om_r, Sx_r, g=g_t, mu=mu_t, nu=nu_t)
    e2 = core.mqm_quantum_energy(y_rot, Om_rot, Sx_rot, g=g_t, mu=mu_t, nu=nu_t)
    Etn1, _, f1, Em1, _ = core.bdg_link_sector(Y_r, kappa=0.2, g1=1.0, g2=0.2)
    Etn2, _, f2, Em2, _ = core.bdg_link_sector(Y_rot, kappa=0.2, g1=1.0, g2=0.2)
    Stn1, Stn2 = core.bdg_subregion_entropy(Em1, f1, Q_0), core.bdg_subregion_entropy(Em2, f2, Q_rot)
    cas1, sb1 = core.gauss_casimir_and_boson_entropy(Y_r, c1, Q_0, full_casimir=True)
    cas2, sb2 = core.gauss_casimir_and_boson_entropy(Y_rot, c2, Q_rot, full_casimir=True)
    y_pt = torch.tensor([0.3, -0.4, 0.5], dtype=core.rdtype, device=device)
    lp1, _ = core.point_probe_dirac(Y_r, y_pt)
    lp2, _ = core.point_probe_dirac(Y_rot, y_pt)
    max_g7 = max(abs(e1["E_mqm"] - e2["E_mqm"]), abs(e1["E_ym4"] - e2["E_ym4"]), abs(float(Etn1 - Etn2)),
                 abs(float(Stn1 - Stn2)), abs(float(cas1 - cas2)), abs(float(sb1 - sb2)), abs(lp1 - lp2))
    record("T06", "U(N) Gauge Invariance (All 7 Observables)", max_g7 < 1e-11, f"MaxErr={max_g7:.2e}")

    # T07: Exact 256D Fock-Space ED (N=2) vs Peschel BdG S_TN
    core2 = GenesisQuantumCore(N=2, D=3, guard=guard)
    Y2 = core2.vec_to_mats(torch.randn(core2.nB, generator=gen, dtype=core2.rdtype).to(device) * 0.9)
    Etn2_ed, _, f2_ed, Em2_ed, _ = core2.bdg_link_sector(Y2, kappa=0.7, g1=0.6, g2=0.1)
    Stn_p = float(core2.bdg_subregion_entropy(Em2_ed, f2_ed, torch.eye(2, dtype=core2.cdtype, device=device)[:, :1]))
    dim_f = 256
    c_ops = []
    for m_i in range(8):
        op = np.zeros((dim_f, dim_f), dtype=np.complex128)
        bit = 1 << m_i
        for st in range(dim_f):
            if st & bit:
                op[st ^ bit, st] = -1.0 if (bin(st & (bit - 1)).count("1") % 2 == 1) else 1.0
        c_ops.append(op)
    Enp, fnp = Em2_ed.cpu().numpy().reshape(4, 4), f2_ed.cpu().numpy()
    Fe = (Enp.T * fnp) @ Enp.conj()
    Hf = np.zeros((dim_f, dim_f), dtype=np.complex128)
    for so in (0, 1):
        for Ie in range(4):
            for Je in range(4):
                if abs(Fe[Ie, Je]) > 1e-15:
                    Hf += 0.7 * Fe[Ie, Je] * (c_ops[2 * Ie + so].conj().T @ c_ops[2 * Je + so])
    for Ie, Jt in enumerate([0, 2, 1, 3]):
        Hf += 1j * 0.7 * (c_ops[2 * Ie].conj().T @ c_ops[2 * Jt + 1].conj().T - c_ops[2 * Jt + 1] @ c_ops[2 * Ie])
    ef, vf = np.linalg.eigh(0.5 * (Hf + Hf.conj().T))
    pt_r = vf[:, 0].reshape(64, 4)
    ev_r2 = np.linalg.eigvalsh(0.5 * (pt_r.T @ pt_r.conj() + (pt_r.T @ pt_r.conj()).conj().T))
    ev_r2 = ev_r2[ev_r2 > 1e-15]
    S_ed = float(-np.sum(ev_r2 * np.log(ev_r2)))
    dE_ed, dS_ed = abs(float(ef[0]) - float(Etn2_ed)), abs(S_ed - Stn_p)
    record("T07", "Exact 256D Fock ED (N=2) vs Peschel BdG", dE_ed < 1e-12 and dS_ed < 1e-12, f"dE={dE_ed:.2e}, dS={dS_ed:.2e}")

    # T08: exact_grad_y vs Finite Differences
    g_an, _ = core.exact_grad_y(y_r, Sx_r, g=g_t, mu=mu_t, nu=nu_t, kappa=0.2, g1=0.5, g2=0.1)
    eps = 1e-6
    g_fd = torch.zeros_like(g_an)
    for k in range(core.nB):
        yp = y_r.clone(); yp[k] += eps
        ym = y_r.clone(); ym[k] -= eps
        ep = core.mqm_quantum_energy(yp, Om_r, Sx_r, g=g_t, mu=mu_t, nu=nu_t)["E_mqm"] + float(core.bdg_link_sector(core.vec_to_mats(yp), kappa=0.2, g1=0.5, g2=0.1)[0])
        em = core.mqm_quantum_energy(ym, Om_r, Sx_r, g=g_t, mu=mu_t, nu=nu_t)["E_mqm"] + float(core.bdg_link_sector(core.vec_to_mats(ym), kappa=0.2, g1=0.5, g2=0.1)[0])
        g_fd[k] = (ep - em) / (2.0 * eps)
    err_g = float(torch.linalg.norm(g_an - g_fd) / torch.clamp(torch.linalg.norm(g_fd), min=1e-12))
    record("T08", "Analytical exact_grad_y vs Finite Diff", err_g < 1e-8, f"RelErr={err_g:.2e}")

    # T09: Provost-Vallee Exact Perturbation vs FD
    g_pv_ex, gap_pv = core.provost_vallee_metric_exact(Y_r, y_pt)
    _, psi0 = core.point_probe_dirac(Y_r, y_pt)
    dpsi = []
    for a in range(3):
        st = torch.zeros(3, dtype=core.rdtype, device=device); st[a] = 1e-5
        _, pp = core.point_probe_dirac(Y_r, y_pt + st)
        _, pm = core.point_probe_dirac(Y_r, y_pt - st)
        pp = pp * (torch.vdot(psi0, pp).conj() / torch.clamp(torch.abs(torch.vdot(psi0, pp)), min=1e-12))
        pm = pm * (torch.vdot(psi0, pm).conj() / torch.clamp(torch.abs(torch.vdot(psi0, pm)), min=1e-12))
        dpsi.append((pp - pm) / 2e-5)
    g_pv_fd = np.array([[float(torch.real(torch.vdot(dpsi[a], dpsi[b]) - torch.vdot(dpsi[a], psi0) * torch.vdot(psi0, dpsi[b]))) for b in range(3)] for a in range(3)])
    err_pv = float(np.max(np.abs(g_pv_ex - 0.5 * (g_pv_fd + g_pv_fd.T))))
    record("T09", "Provost-Vallee Exact PT vs Finite Diff", err_pv < 1e-7, f"MaxErr={err_pv:.2e} (gap={gap_pv:.2f})")

    # T10, T11, T12: Stage 3 Adjoint Dirac Matter & Singularity-Free Daleckii-Krein Gradient
    matter = AdjointDiracMatterSector(core, m_psi=0.5)
    Dm = matter.build_dirac_matrix(y_r)
    h_err = float(torch.linalg.norm(Dm - Dm.conj().T))
    ev_d = torch.linalg.eigvalsh(Dm)
    sym_err = float(torch.max(torch.abs(ev_d + torch.flip(ev_d, dims=[0]))))
    record("T10", "Adjoint Dirac Hermiticity & +-E Symmetry", h_err < 1e-13 and sym_err < 1e-12, f"Herm={h_err:.1e}, Sym={sym_err:.2e}")

    rho1, cm1 = matter.build_localized_probe_packet(Y_r, y_pt)
    rho2, cm2 = matter.build_localized_probe_packet(Y_rot, y_pt)
    Em1_v, gm1 = matter.matter_energy_and_grad(y_r, cm1, 5.0)
    Em2_v, _ = matter.matter_energy_and_grad(y_rot, cm2, 5.0)
    cov_err, dEm_err = float(torch.linalg.norm(rho2 - U_h @ rho1 @ U_h.conj().T)), abs(Em1_v - Em2_v)
    record("T11", "Adjoint Dirac U(N) Gauge Invariance", cov_err < 1e-12 and dEm_err < 1e-12, f"Cov={cov_err:.2e}, dE={dEm_err:.2e}")

    L4 = build_su2_irrep(N, device)
    y_sph = core.mats_to_vec(2.0 * L4)
    _, c_sph = matter.build_localized_probe_packet(2.0 * L4, torch.tensor([0.0, 0.0, 3.0], dtype=core.rdtype, device=device))
    _, g_sph = matter.matter_energy_and_grad(y_sph, c_sph, 5.0)
    fd_errs = []
    for k in range(0, core.nB, 4):
        yp = y_sph.clone(); yp[k] += 1e-6
        ym = y_sph.clone(); ym[k] -= 1e-6
        fd_errs.append(abs((matter.matter_energy_only(yp, c_sph, 5.0) - matter.matter_energy_only(ym, c_sph, 5.0)) / 2e-6 - float(g_sph[k])))
    rel_dk = max(fd_errs) / max(float(torch.max(torch.abs(g_sph))), 1e-12)
    record("T12", "Daleckii-Krein Grad on Degenerate Sphere", (not torch.isnan(g_sph).any()) and rel_dk < 1e-7, f"RelErr={rel_dk:.2e}")

    # T13 & T14: Phase 3B O(N^3) Matrix-Native Core Exact Match (Y=0 & Y!=0 Hessian Trace) & Envelope Gradient
    native = MatrixNativeQuantumCore(N=N, D=3, guard=guard)
    Ecl_n, _ = native.classical_energy_and_grad(Y_r, g=g_t, mu=mu_t, nu=nu_t)
    Ecl_ref = e1["E_cl"] + 0.5 * mu_t * mu_t * float(torch.sum(y_r * y_r))
    qn = native.solve_1loop_hfb_and_bdg(Y_r, g=g_t, mu=mu_t, kappa=0.2, g1=1.0, g2=0.2, compute_grad=False)
    Om_z, Sx_z, _, _ = core.solve_hfb_omega(y_zero, g=g_t, mu=mu_t, nu=0.0, max_iter=30, tol=1e-13)
    ez_ref = core.mqm_quantum_energy(y_zero, Om_z, Sx_z, g=g_t, mu=mu_t, nu=0.0)["E_mqm"]
    ez_nat = native.solve_1loop_hfb_and_bdg(core.vec_to_mats(y_zero), g=g_t, mu=mu_t, kappa=0.0, compute_grad=False)["E_1loop"]
    # Non-zero Y audit: exact trace identity Tr(H2_YM) == D * ((D-1)/D) * g^2 * Tr(Delta_Y)
    H2_ref, _ = core.build_h2_and_kloop(y_r, Sx_z, g=g_t, mu=mu_t, nu=nu_t)
    tr_h2_ym_ref = float(torch.trace(H2_ref) - core.nB * (mu_t * mu_t))
    tr_h2_ym_nat = float(3.0 * (2.0 / 3.0) * (g_t * g_t) * torch.sum(qn["lam"]))
    err_tr = abs(tr_h2_ym_ref - tr_h2_ym_nat) / max(abs(tr_h2_ym_ref), 1e-12)
    err_nat = max(abs(Ecl_n - Ecl_ref), abs(qn["E_tn"] - float(Etn1)), abs(ez_nat - ez_ref), err_tr)
    record("T13", "O(N^3) Engine vs HFB (Y=0) & Y!=0 Tr(H2)", err_nat < 1e-10, f"MaxErr={err_nat:.2e} (TrH2={err_tr:.2e})")

    En1, Gn1, _ = native.total_energy_and_grad(Y_r, g=g_t, mu=mu_t, nu=nu_t, kappa=0.2, g1=0.5, g2=0.1)
    En2, _, _ = native.total_energy_and_grad(Y_rot, g=g_t, mu=mu_t, nu=nu_t, kappa=0.2, g1=0.5, g2=0.1, compute_grad=False)
    dY = native.random_isotropic_gue(1.0, gen)
    dY = dY / torch.sqrt(torch.real(torch.sum(dY * dY.conj())))
    fd_n = (native.total_energy_and_grad(Y_r + 1e-6 * dY, g=g_t, mu=mu_t, nu=nu_t, kappa=0.2, g1=0.5, g2=0.1, compute_grad=False)[0]
            - native.total_energy_and_grad(Y_r - 1e-6 * dY, g=g_t, mu=mu_t, nu=nu_t, kappa=0.2, g1=0.5, g2=0.1, compute_grad=False)[0]) / 2e-6
    an_n = float(torch.real(torch.sum(dY * Gn1.conj())))
    err_ng = abs(fd_n - an_n) / max(abs(fd_n), 1e-12)
    record("T14", "O(N^3) U(N) Invariance & Envelope Grad", abs(En1 - En2) < 1e-11 and err_ng < 1e-7, f"dE={abs(En1-En2):.2e}, Grad={err_ng:.2e}")

    # T15: Phase 3C 4D Gauge-Higgs Hessian U(6) Invariance
    L3 = build_su2_irrep(3, device)
    T6 = build_hermitian_basis(6, device)
    Y4_t = torch.zeros((4, 6, 6), dtype=torch.complex128, device=device)
    for i in range(3): Y4_t[i] = torch.kron(torch.eye(2, dtype=torch.complex128, device=device), 5.0 * L3[i])
    Y4_t[3] = 2.0 * torch.kron(torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device), torch.eye(3, dtype=torch.complex128, device=device))
    U6 = _rand_haar(6, gen, device)
    Y4_r = torch.einsum("ij,djk,lk->dil", U6, Y4_t, U6.conj()).contiguous()
    H4a, _ = build_exact_4d_hessian_matrix_free(Y4_t, T6)
    H4b, _ = build_exact_4d_hessian_matrix_free(Y4_r, T6)
    err_h4 = float(torch.max(torch.abs(torch.linalg.eigvalsh(H4a) - torch.linalg.eigvalsh(H4b))))
    record("T15", "4D Gauge-Higgs Hessian U(6) Invariance", err_h4 < 1e-11, f"MaxErr={err_h4:.2e}")

    # T16: Phase 3D 6D Clifford Algebra {Gamma_I, Gamma_J} = 2 delta_{IJ} I_8
    gam6 = build_6d_clifford_gammas(device)
    id8 = torch.eye(8, dtype=torch.complex128, device=device)
    cl_err = max(float(torch.linalg.norm(gam6[i] @ gam6[j] + gam6[j] @ gam6[i] - (2.0 if i == j else 0.0) * id8)) for i in range(6) for j in range(6))
    record("T16", "6D Clifford Algebra {Gamma_I, Gamma_J}", cl_err < 1e-13, f"MaxErr={cl_err:.2e}")

    # T17: Phase 3D GKP Chiral Dirac Gauge Invariance, {D, g5}=0, & Noisy Atiyah-Singer Index q=3
    L4_c, L7_c = 10.0 * build_su2_irrep(4, device), 10.0 * build_su2_irrep(7, device)
    Dc1, g5_1 = build_exact_gkp_chiral_dirac(L4_c, L7_c, 10.0)
    U4c, U7c = _rand_haar(4, gen, device), _rand_haar(7, gen, device)
    Dc2, _ = build_exact_gkp_chiral_dirac(torch.einsum("ij,djk,lk->dil", U4c, L4_c, U4c.conj()).contiguous(),
                                          torch.einsum("ij,djk,lk->dil", U7c, L7_c, U7c.conj()).contiguous(), 10.0)
    g_ch = float(torch.max(torch.abs(torch.linalg.eigvalsh(Dc1) - torch.linalg.eigvalsh(Dc2))))
    ac_ch = float(torch.linalg.norm(Dc1 @ g5_1 + g5_1 @ Dc1))
    # Add random Hermitian noise to both branes and confirm exact Index = +3 stays pinned at E=0!
    Z4 = (torch.randn((3, 4, 4), generator=gen, dtype=torch.float64) + 1j * torch.randn((3, 4, 4), generator=gen, dtype=torch.float64)).to(device)
    Z7 = (torch.randn((3, 7, 7), generator=gen, dtype=torch.float64) + 1j * torch.randn((3, 7, 7), generator=gen, dtype=torch.float64)).to(device)
    Dc_n, g5_n = build_exact_gkp_chiral_dirac(L4_c + 0.6 * (Z4 + Z4.conj().transpose(-2, -1)), L7_c + 0.6 * (Z7 + Z7.conj().transpose(-2, -1)), 10.0)
    ew_n, ev_n = torch.linalg.eigh(Dc_n)
    zm = torch.abs(ew_n) < 1e-10
    Vz = ev_n[:, zm]
    g5_ev = torch.real(torch.linalg.eigvalsh(0.5 * (Vz.conj().T @ g5_n @ Vz + (Vz.conj().T @ g5_n @ Vz).conj().T)))
    idx_ch = int(torch.sum(g5_ev > 0.5) - torch.sum(g5_ev < -0.5))
    record("T17", "GKP Chiral Dirac Gauge, {D,g5}=0 & Index=3", g_ch < 1e-12 and ac_ch < 1e-12 and idx_ch == 3,
           f"Gauge={g_ch:.1e}, {{D,g5}}={ac_ch:.1e}, NoisyIdx={idx_ch:+d}")

    # T18: Level-2 General Dirty-Law U(N) Gauge Invariance & Analytical Gradient
    from .experiments.law_ecosystem import GeneralRandomMatrixLaw, _random_spd_3x3, _random_sym_cubic_3x3x3
    rng_l2 = np.random.default_rng(9001)
    law_t = GeneralRandomMatrixLaw(
        M=_random_spd_3x3(rng_l2), S_sym=_random_sym_cubic_3x3x3(rng_l2, 0.3),
        nu_flux=0.75, g_ym=np.array([0.20, 0.35, 0.28]),
        q_diag=np.array([0.05, 0.08, 0.03]), r_cross=np.array([0.04, 0.06, 0.02]),
        device=device,
    )
    El2_1, Gl2_1 = law_t.energy_and_grad(Y_r, compute_grad=True)
    El2_2, _ = law_t.energy_and_grad(Y_rot, compute_grad=False)
    g_l2_err = abs(El2_1 - El2_2) / max(abs(El2_1), 1.0)
    fd_l2 = (law_t.energy_and_grad(Y_r + 1e-5 * dY, False)[0] - law_t.energy_and_grad(Y_r - 1e-5 * dY, False)[0]) / 2e-5
    an_l2 = float(torch.real(torch.sum(dY * Gl2_1.conj())))
    gr_l2_err = abs(fd_l2 - an_l2) / max(abs(fd_l2), 1e-12)
    record("T18", "Dirty-Law U(N) Gauge Inv & Analytical Grad", g_l2_err < 1e-12 and gr_l2_err < 1e-6,
           f"Gauge={g_l2_err:.2e}, Grad={gr_l2_err:.2e}")

    # T19: Level-2 Bootstrap Exact Loop Equations (1D AHO & Large-N 1-Matrix QM)
    from .experiments.matrix_bootstrap import (
        exact_1d_aho_fock, exact_large_n_1matrix_bipz,
        bootstrap_1d_aho_moments, bootstrap_1matrix_moments,
        _hankel_min_eig_scaled, _matrix1_min_eig_with_p,
        bootstrap_savvidy_ym_sdp,
    )
    E_aho_t, x2_aho_t = exact_1d_aho_fock(g=1.0, n_fock=60)
    lam_aho_t = _hankel_min_eig_scaled(bootstrap_1d_aho_moments(E_aho_t, x2_aho_t, 1.0, 10), 10)
    e_1m_t, t2_1m_t, _ = exact_large_n_1matrix_bipz(g=1.0)
    lam_1m_t = _matrix1_min_eig_with_p(bootstrap_1matrix_moments(e_1m_t, t2_1m_t, 1.0, 10), e_1m_t, 1.0, 10)
    record("T19", "Bootstrap Loop Eqs (1D AHO & Large-N 1M)", lam_aho_t >= -1e-9 and lam_1m_t >= -1e-9,
           f"1D_eig={lam_aho_t:+.2e}, 1M_eig={lam_1m_t:+.2e}")

    # T20: Level-2 Symbolic Weyl-Algebra Savvidy Yang-Mills SDP Sandwich (g^2 = 1.0)
    sdp_t = bootstrap_savvidy_ym_sdp(g2=1.0)
    sdp_ok = (
        sdp_t["E_boot_lower"] <= sdp_t["E_exact"] + 1e-6 <= sdp_t["E_hfb_upper"] + 1e-6
        and sdp_t["boot_err_pct"] < 0.2
        and sdp_t["duality_gap"] < 1e-7
    )
    record("T20", "Savvidy YM Symbolic SDP Sandwich (g^2=1)", sdp_ok,
           f"Boot={sdp_t['E_boot_lower']:.5f}<=Ex={sdp_t['E_exact']:.5f}<=HFB={sdp_t['E_hfb_upper']:.5f} (KKT={sdp_t['duality_gap']:.1e})")

    n_pass = sum(1 for r in results.values() if r["passed"])
    n_tot = len(results)
    ram_mb, vram_mb = guard.check_runtime()
    elapsed = time.time() - t0
    print("-" * 92)
    print(f"[MASTER AUDIT] {n_pass}/{n_tot} TESTS PASSED | Elapsed: {elapsed:.2f}s | RAM: {ram_mb:.1f} MB | VRAM: {vram_mb:.1f} MB")
    print("=" * 92)
    guard.cleanup()
    return {"passed_count": n_pass, "total_count": n_tot, "all_passed": n_pass == n_tot, "elapsed_sec": elapsed, "ram_mb": ram_mb, "vram_mb": vram_mb, "tests": results}
