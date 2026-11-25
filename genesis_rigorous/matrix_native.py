"""
O(N^3) Matrix-Native Quantum Engine (ZERO O(N^6) f_abc allocation).
Supports arbitrary spatial dimension D (D=3 fuzzy S^2, D=6 4D S^2 x S^2) and
large matrix sizes N up to 32+ within < 100 MB VRAM on consumer GPUs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .safety import HardwareSafetyGuard, SafetyConfig, safe_eigh, safe_eigvalsh


class MatrixNativeQuantumCore:
    """
    O(N^3) commutator engine + O(N^4) Kronecker Noncommutative Laplacian
    + Exact Self-Consistent 1-Loop Open-String HFB (decoupled U(1) trace + SU(N) gap)
    + Exact 2-Flavor BdG Link-Fermion Backreaction with Envelope-Theorem Matrix Gradient.
    """

    def __init__(self, N: int, D: int = 3, guard: Optional[HardwareSafetyGuard] = None):
        self.guard = guard or HardwareSafetyGuard(SafetyConfig())
        self.preflight_info = self.guard.preflight_check(N=N, D=D, mode="matrix_native")
        self.device = self.guard.device
        self.N = N
        self.D = D
        self.N2 = N * N
        self.rdtype = torch.float64
        self.cdtype = torch.complex128

        self.eye_N = torch.eye(N, dtype=self.cdtype, device=self.device)
        self.sigma = torch.stack([
            torch.tensor([[0, 1], [1, 0]], dtype=self.cdtype),
            torch.tensor([[0, -1j], [1j, 0]], dtype=self.cdtype),
            torch.tensor([[1, 0], [0, -1]], dtype=self.cdtype),
        ]).to(self.device)
        self.eps3_pairs = [
            (0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0),
            (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0),
        ]

    def project_hermitian_traceless(self, Y: torch.Tensor) -> torch.Tensor:
        Yh = 0.5 * (Y + Y.conj().transpose(-2, -1))
        tr = torch.einsum("dii->d", Yh) / float(self.N)
        return Yh - tr.view(-1, 1, 1) * self.eye_N.unsqueeze(0)

    def random_isotropic_gue(self, y_scale: float, gen: torch.Generator) -> torch.Tensor:
        """100% U(N) x SO(D) isotropic traceless Hermitian GUE random matrices."""
        re = torch.randn((self.D, self.N, self.N), generator=gen, dtype=self.rdtype)
        im = torch.randn((self.D, self.N, self.N), generator=gen, dtype=self.rdtype)
        Z = (re + 1j * im).to(self.device)
        H = 0.5 * (Z + Z.conj().transpose(-2, -1)) * float(y_scale)
        return self.project_hermitian_traceless(H)

    def classical_energy_and_grad(
        self, Y: torch.Tensor, g: float = 0.25, mu: float = 1.0, nu: float = 3.0,
        flux_triples: Optional[List[Tuple[int, int, int, float]]] = None
    ) -> Tuple[float, torch.Tensor]:
        g2 = g * g
        E_mass = 0.5 * (mu * mu) * float(torch.real(torch.sum(Y * Y.conj())))
        grad = (mu * mu) * Y.clone()

        E_ym = 0.0
        for I in range(self.D):
            for J in range(I + 1, self.D):
                C_IJ = Y[I] @ Y[J] - Y[J] @ Y[I]
                E_ym += 0.5 * g2 * float(torch.real(torch.sum(C_IJ * C_IJ.conj())))
                grad[I] += -g2 * (Y[J] @ C_IJ - C_IJ @ Y[J])
                grad[J] += +g2 * (Y[I] @ C_IJ - C_IJ @ Y[I])

        E_flux = 0.0
        triples = flux_triples if flux_triples is not None else (self.eps3_pairs if self.D == 3 else [])
        if nu != 0.0 and len(triples) > 0:
            for I, J, K, sgn in triples:
                C_JK = Y[J] @ Y[K] - Y[K] @ Y[J]
                E_flux += float(torch.real(1j * (nu * g / 3.0) * sgn * torch.einsum("ij,ji->", Y[I], C_JK)))
                grad[I] += (1j * nu * g * sgn) * C_JK

        return E_mass + E_ym + E_flux, self.project_hermitian_traceless(grad)

    def build_laplacian_mat(self, Y: torch.Tensor) -> torch.Tensor:
        """Exact Kronecker Noncommutative Laplacian: Delta_Y = sum_I (Y_I^2 (x) I + I (x) (Y_I^2)^T - 2 Y_I (x) Y_I^T)."""
        Delta = torch.zeros((self.N2, self.N2), dtype=self.cdtype, device=self.device)
        for I in range(self.D):
            YI = Y[I].contiguous()
            YI2 = (YI @ YI).contiguous()
            Delta += (
                torch.kron(YI2, self.eye_N)
                + torch.kron(self.eye_N, YI2.T.contiguous())
                - 2.0 * torch.kron(YI, YI.T.contiguous())
            )
        return 0.5 * (Delta + Delta.conj().T)

    def solve_1loop_hfb_and_bdg(
        self, Y: torch.Tensor, m: float = 1.0, g: float = 0.25, mu: float = 1.0,
        kappa: float = 0.15, Nf: int = 1, g1: float = 0.015, g2: float = 0.0001,
        compute_grad: bool = True
    ) -> Dict[str, object]:
        Delta = self.build_laplacian_mat(Y)
        if compute_grad:
            lam, U_eig = safe_eigh(Delta)
        else:
            lam = safe_eigvalsh(Delta)
            U_eig = None
        lam = torch.clamp(lam, min=0.0)

        # Separate decoupled k=0 U(1) trace mode from k=1..N^2-1 interacting SU(N) modes.
        # Exact diagonal YM Hessian block is H2_{II} = mu^2 I + g^2 sum_{J!=I} ad_{Y_J}^2,
        # whose isotropic average across I=1..D carries exact factor (D-1)/D on Delta_Y.
        lam_su = lam[1:]
        g2v = g * g
        c_lap = float(self.D - 1) / float(self.D)
        coeff_h = 2.0 * g2v * (self.D - 1) * float(self.N)
        M_eff_sq = mu * mu + 1.0
        for _ in range(40):
            om_su = torch.sqrt(torch.clamp(m * (M_eff_sq + c_lap * g2v * lam_su), min=1e-12))
            sig2_su = float(torch.mean(0.5 / om_su))
            M_new = mu * mu + coeff_h * sig2_su
            if abs(M_new - M_eff_sq) < 1e-14 * max(M_new, 1.0):
                M_eff_sq = M_new
                break
            M_eff_sq = 0.7 * M_new + 0.3 * M_eff_sq

        om_su = torch.sqrt(torch.clamp(m * (M_eff_sq + c_lap * g2v * lam_su), min=1e-12))
        om_tr = torch.tensor([np.sqrt(m) * mu], dtype=self.rdtype, device=self.device)
        om = torch.cat([om_tr, om_su], dim=0)
        sig2_su = float(torch.mean(0.5 / om_su))

        E_kin = float(0.25 * self.D / m * torch.sum(om))
        E_fluc_pot = float(0.25 * self.D * torch.sum((mu * mu + c_lap * g2v * lam) / om))
        E_ym4 = 0.5 * g2v * self.D * (self.D - 1) * self.N * (self.N2 - 1) * (sig2_su ** 2)
        E_1loop = E_kin + E_fluc_pot + E_ym4

        f_v = g1 * lam + g2 * (lam * lam)
        E_tn = float(kappa * Nf * torch.sum(f_v - torch.sqrt(f_v * f_v + 1.0)))

        grad_q = None
        if compute_grad:
            dE_bos = torch.zeros_like(lam)
            dE_bos[1:] = 0.25 * self.D * c_lap * g2v / om_su
            dE_dlam = dE_bos + kappa * Nf * (1.0 - f_v / torch.sqrt(f_v * f_v + 1.0)) * (g1 + 2.0 * g2 * lam)
            U_mats = U_eig.T.reshape(self.N2, self.N, self.N)
            U_dag = U_mats.conj().transpose(-2, -1)
            grad_q = torch.zeros_like(Y)
            w_c = dE_dlam.to(self.cdtype).view(-1, 1, 1)
            for I in range(self.D):
                YI = Y[I].unsqueeze(0)
                A_k = YI @ U_mats - U_mats @ YI
                A_dag = A_k.conj().transpose(-2, -1)
                term = (U_mats @ A_dag - A_dag @ U_mats) + (A_k @ U_dag - U_dag @ A_k)
                grad_q[I] = torch.sum(w_c * term, dim=0)
            grad_q = self.project_hermitian_traceless(grad_q)

        return {
            "E_1loop": E_1loop,
            "E_ym4": E_ym4,
            "E_tn": E_tn,
            "M_eff_sq": M_eff_sq,
            "lam": lam,
            "grad_q": grad_q,
        }

    def total_energy_and_grad(
        self, Y: torch.Tensor, g: float = 0.25, mu: float = 1.0, nu: float = 3.0,
        kappa: float = 0.15, Nf: int = 1, g1: float = 0.015, g2: float = 0.0001,
        flux_triples: Optional[List[Tuple[int, int, int, float]]] = None,
        compute_grad: bool = True
    ) -> Tuple[float, Optional[torch.Tensor], Dict[str, object]]:
        E_cl, g_cl = self.classical_energy_and_grad(Y, g=g, mu=mu, nu=nu, flux_triples=flux_triples)
        q_res = self.solve_1loop_hfb_and_bdg(Y, g=g, mu=mu, kappa=kappa, Nf=Nf, g1=g1, g2=g2, compute_grad=compute_grad)
        E_tot = E_cl + q_res["E_1loop"] + q_res["E_tn"]
        grad_tot = (g_cl + q_res["grad_q"]) if compute_grad else None
        q_res["E_cl"] = E_cl
        q_res["E_tot"] = E_tot
        return E_tot, grad_tot, q_res

    @torch.no_grad()
    def point_probe_dirac(self, Y: torch.Tensor, y_pt: torch.Tensor) -> Tuple[float, torch.Tensor]:
        D_mat = torch.zeros((2 * self.N, 2 * self.N), dtype=self.cdtype, device=self.device)
        for I in range(3):
            D_mat += torch.kron((Y[I] - y_pt[I] * self.eye_N).contiguous(), self.sigma[I].contiguous())
        D_mat = 0.5 * (D_mat + D_mat.conj().T)
        evals, evecs = safe_eigh(D_mat)
        idx = int(torch.argmin(torch.abs(evals)))
        return float(torch.abs(evals[idx])), evecs[:, idx]

    @torch.no_grad()
    def provost_vallee_metric_exact(self, Y: torch.Tensor, y_pt: torch.Tensor, deg_tol: float = 1e-9) -> Tuple[np.ndarray, float]:
        D_mat = torch.zeros((2 * self.N, 2 * self.N), dtype=self.cdtype, device=self.device)
        for I in range(3):
            D_mat += torch.kron((Y[I] - y_pt[I] * self.eye_N).contiguous(), self.sigma[I].contiguous())
        D_mat = 0.5 * (D_mat + D_mat.conj().T)
        evals, evecs = safe_eigh(D_mat)
        idx0 = int(torch.argmin(torch.abs(evals)))
        lam0, psi0 = evals[idx0], evecs[:, idx0]
        diffs = evals - lam0
        mask = torch.abs(diffs) > deg_tol
        gap = float(torch.min(torch.abs(diffs[mask]))) if torch.any(mask) else 0.0
        psi_o = evecs[:, mask]
        inv_d = (1.0 / diffs[mask]).to(self.cdtype)
        M_mat = torch.zeros((3, psi_o.shape[1]), dtype=self.cdtype, device=self.device)
        for a in range(3):
            v_a = torch.kron(self.eye_N, self.sigma[a].contiguous()) @ psi0
            M_mat[a] = (psi_o.conj().T @ v_a) * inv_d
        g_m = torch.real(M_mat.conj() @ M_mat.T).cpu().numpy()
        return 0.5 * (g_m + g_m.T), gap
