"""
Exact 4-Component Adjoint Dirac Matter Sector (D_adj(Y)) with Singularity-Free
Daleckii-Krein Rationalized Divided-Difference Gradient + Emergent Meridian
Differential Geometry Scanner (Gaussian Curvature K(theta) & Provost-Vallee Metric).
"""

from typing import Dict, Tuple

import numpy as np
import torch

from .exact_hfb import GenesisQuantumCore
from .safety import safe_eigh


class AdjointDiracMatterSector:
    """
    Constructs the exact 4*N^2 x 4*N^2 Hermitian single-particle adjoint Dirac operator:
      D_adj(Y) = sum_{I=1}^3 alpha^I (x) (-i F_mat[I]) + m_psi (beta (x) I_{N^2})
    Evaluates the spin-averaged positive-energy expectation value:
      E_matter(Y) = M_src * (1/4) sum_{s=1}^4 <e_s (x) c | sqrt(D_adj(Y)^2) | e_s (x) c>
    and its exact Daleckii-Krein gradient (singularity-free even on exact SO(3) spheres).
    """

    def __init__(self, core: GenesisQuantumCore, m_psi: float = 0.5):
        self.core = core
        self.m_psi = float(m_psi)
        self.device = core.device
        self.rdtype = core.rdtype
        self.cdtype = core.cdtype

        tau_x = torch.tensor([[0, 1], [1, 0]], dtype=self.cdtype, device=self.device)
        tau_z = torch.tensor([[1, 0], [0, -1]], dtype=self.cdtype, device=self.device)
        eye2 = torch.eye(2, dtype=self.cdtype, device=self.device)

        self.alpha = torch.stack([
            torch.kron(tau_x, core.sigma[I].contiguous()) for I in range(3)
        ], dim=0)
        self.beta = torch.kron(tau_z, eye2)
        self.eye_N2_c = torch.eye(core.N2, dtype=self.cdtype, device=self.device)

    def build_dirac_matrix(self, y_vec: torch.Tensor) -> torch.Tensor:
        y = y_vec.reshape(3, self.core.N2)
        F_mat = (y @ self.core.f_flat).reshape(3, self.core.N2, self.core.N2)
        A_mats = (-1j) * F_mat.to(self.cdtype)
        D_4 = torch.einsum("isp,iab->sapb", self.alpha, A_mats)
        D_4 = D_4 + self.m_psi * torch.einsum("sp,ab->sapb", self.beta, self.eye_N2_c)
        D_mat = D_4.reshape(4 * self.core.N2, 4 * self.core.N2)
        return 0.5 * (D_mat + D_mat.conj().T)

    @torch.no_grad()
    def build_localized_probe_packet(self, Y: torch.Tensor, y0_pt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """100% U(N)-covariant spatial coherent packet rho_{y0} = Tr_spin(|psi_0(y0)><psi_0(y0)|)."""
        _, psi0 = self.core.point_probe_dirac(Y, y0_pt)
        psi_mat = psi0.reshape(self.core.N, 2)
        rho = psi_mat @ psi_mat.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        norm_f = torch.sqrt(torch.real(torch.sum(rho * rho.conj())))
        rho = rho / torch.clamp(norm_f, min=1e-12)
        c_vec = torch.real(torch.einsum("ij,aji->a", rho, self.core.T_basis))
        c_vec = c_vec / torch.clamp(torch.linalg.norm(c_vec), min=1e-12)
        return rho, c_vec

    @torch.no_grad()
    def matter_energy_and_grad(
        self, y_vec: torch.Tensor, c_vec: torch.Tensor, M_src: float
    ) -> Tuple[float, torch.Tensor]:
        """Exact Daleckii-Krein rationalized divided-difference energy and gradient."""
        if M_src == 0.0:
            return 0.0, torch.zeros_like(y_vec)

        D_mat = self.build_dirac_matrix(y_vec)
        evals, evecs = safe_eigh(D_mat)
        abs_e = torch.sqrt(evals * evals + 1e-24)

        V_s = evecs.reshape(4, self.core.N2, 4 * self.core.N2)
        proj = torch.einsum("a,san->sn", c_vec.to(self.cdtype), V_s)
        weights = torch.real(proj.conj() * proj)
        E_mat = float(M_src) * float(0.25 * torch.sum(weights * abs_e.unsqueeze(0)))

        R_eig = 0.25 * (proj.conj().T @ proj)
        F1 = (evals.unsqueeze(1) + evals.unsqueeze(0)) / (abs_e.unsqueeze(1) + abs_e.unsqueeze(0))
        W_mat = (evecs @ (F1.to(self.cdtype) * R_eig)) @ evecs.conj().T
        W_4 = W_mat.reshape(4, self.core.N2, 4, self.core.N2)
        M_I_real = torch.real(torch.einsum("pbsa,isp->iab", W_4, self.alpha) * (-1j))
        grad_y = float(M_src) * (M_I_real.reshape(3, self.core.N2 * self.core.N2) @ self.core.f_flat.T)
        return E_mat, grad_y.reshape(-1)

    @torch.no_grad()
    def matter_energy_only(self, y_vec: torch.Tensor, c_vec: torch.Tensor, M_src: float) -> float:
        if M_src == 0.0:
            return 0.0
        D_mat = self.build_dirac_matrix(y_vec)
        evals, evecs = safe_eigh(D_mat)
        abs_e = torch.sqrt(evals * evals + 1e-24)
        V_s = evecs.reshape(4, self.core.N2, 4 * self.core.N2)
        proj = torch.einsum("a,san->sn", c_vec.to(self.cdtype), V_s)
        weights = torch.real(proj.conj() * proj)
        return float(M_src) * float(0.25 * torch.sum(weights * abs_e.unsqueeze(0)))


@torch.no_grad()
def find_shell_radius_along_ray(
    core, Y: torch.Tensor, direction: np.ndarray, r_hint: float
) -> Tuple[float, float]:
    """Finds exact Point-Probe Dirac zero-mode shell radius r*(n_hat) via coarse + golden search."""
    d_t = torch.tensor(direction / np.linalg.norm(direction), dtype=core.rdtype, device=core.device)
    r_lo = max(0.35 * r_hint, 0.5)
    r_hi = 1.55 * r_hint
    r_grid = np.linspace(r_lo, r_hi, 65)
    vals = [core.point_probe_dirac(Y, float(r) * d_t)[0] for r in r_grid]
    idx = int(np.argmin(vals))
    a = float(r_grid[max(0, idx - 1)])
    b = float(r_grid[min(len(r_grid) - 1, idx + 1)])
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(24):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        if core.point_probe_dirac(Y, c * d_t)[0] < core.point_probe_dirac(Y, d * d_t)[0]:
            b = d
        else:
            a = c
    r_star = 0.5 * (a + b)
    return r_star, core.point_probe_dirac(Y, r_star * d_t)[0]


@torch.no_grad()
def scan_meridian_geometry(
    core, Y: torch.Tensor, r_hint: float, n_theta: int = 37
) -> Dict[str, object]:
    """Scans polar angle theta in [0, pi] extracting r_shell(theta), Provost-Vallee g_ab, and intrinsic K(theta)."""
    thetas = np.linspace(0.0, np.pi, n_theta)
    r_arr = np.zeros(n_theta, dtype=float)
    lmin_arr = np.zeros(n_theta, dtype=float)
    g_tr_arr = np.zeros(n_theta, dtype=float)
    g_area_arr = np.zeros(n_theta, dtype=float)
    g_evals_arr = np.zeros((n_theta, 3), dtype=float)

    for i, th in enumerate(thetas):
        n_hat = np.array([np.sin(th), 0.0, np.cos(th)], dtype=float)
        r_s, l_s = find_shell_radius_along_ray(core, Y, n_hat, r_hint=r_hint)
        r_arr[i] = r_s
        lmin_arr[i] = l_s
        y_pt = torch.tensor(r_s * n_hat, dtype=core.rdtype, device=core.device)
        g_ab, _ = core.provost_vallee_metric_exact(Y, y_pt)
        ev = np.sort(np.linalg.eigvalsh(g_ab))
        g_evals_arr[i] = ev
        g_tr_arr[i] = float(ev[1] + ev[2])
        g_area_arr[i] = float(np.sqrt(max(ev[1], 0.0) * max(ev[2], 0.0)))

    # Smooth axisymmetric cosine-series fit for noise-free intrinsic Gaussian curvature K(theta)
    L_fit = min(8, n_theta // 4)
    A_cos = np.column_stack([np.cos(l * thetas) for l in range(L_fit + 1)])
    coeffs, _, _, _ = np.linalg.lstsq(A_cos, r_arr, rcond=None)
    r_sm = A_cos @ coeffs
    dr = np.column_stack([-l * np.sin(l * thetas) for l in range(L_fit + 1)]) @ coeffs
    d2r = np.column_stack([-(l ** 2) * np.cos(l * thetas) for l in range(L_fit + 1)]) @ coeffs

    K_gauss = np.zeros(n_theta, dtype=float)
    for i, th in enumerate(thetas):
        rv, dv, d2v = r_sm[i], dr[i], d2r[i]
        denom_sq = rv * rv + dv * dv
        k_th = (rv * rv + 2.0 * dv * dv - rv * d2v) / (denom_sq ** 1.5)
        k_ph = k_th if np.sin(th) < 1e-6 else (rv - dv * (np.cos(th) / np.sin(th))) / (rv * np.sqrt(denom_sq))
        K_gauss[i] = k_th * k_ph

    return {
        "thetas": thetas.tolist(),
        "r_shell": r_arr.tolist(),
        "r_smooth": r_sm.tolist(),
        "lmin": lmin_arr.tolist(),
        "g_tang_tr": g_tr_arr.tolist(),
        "sqrt_det_g": g_area_arr.tolist(),
        "g_evals": g_evals_arr.tolist(),
        "K_gauss": K_gauss.tolist(),
        "r_north": float(r_arr[0]),
        "r_equator": float(r_arr[n_theta // 2]),
        "r_south": float(r_arr[-1]),
        "K_north": float(K_gauss[0]),
        "K_equator": float(K_gauss[n_theta // 2]),
        "K_south": float(K_gauss[-1]),
        "g_area_north": float(g_area_arr[0]),
        "g_area_south": float(g_area_arr[-1]),
    }
