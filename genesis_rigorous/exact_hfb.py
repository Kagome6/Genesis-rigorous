"""
U(N)-Covariant Mean-Field HFB Gaussian Matrix Wave-Packet Core + 2-Flavor BdG Link Fermions
(including Traceless su(N) k=0-Projected Link Sector) + Coulomb-Transverse Projected Casimir
+ Cholesky Bosonic Symplectic Entropy + Point-Probe Dirac Geometry & Exact Provost-Vallee Metric.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .safety import HardwareSafetyGuard, safe_eigh, safe_eigvalsh


def build_hermitian_basis(N: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Orthonormal Hermitian matrix basis {T^A}_{A=1}^{N^2} satisfying Tr(T^A T^B) = delta_{AB}."""
    basis = []
    for a in range(N):
        T = torch.zeros((N, N), dtype=torch.complex128)
        T[a, a] = 1.0
        basis.append(T)
    inv_s2 = 1.0 / np.sqrt(2.0)
    for a in range(N):
        for b in range(a + 1, N):
            T = torch.zeros((N, N), dtype=torch.complex128)
            T[a, b] = inv_s2
            T[b, a] = inv_s2
            basis.append(T)
    for a in range(N):
        for b in range(a + 1, N):
            T = torch.zeros((N, N), dtype=torch.complex128)
            T[a, b] = 1j * inv_s2
            T[b, a] = -1j * inv_s2
            basis.append(T)
    out = torch.stack(basis, dim=0)
    return out.to(device) if device is not None else out


def build_su2_irrep(n: int, device: torch.device) -> torch.Tensor:
    """Exact n-dimensional spin j=(n-1)/2 Hermitian SU(2) generators L_1, L_2, L_3 (diagnostic/reference)."""
    j = 0.5 * (n - 1)
    m_vals = np.arange(j, -j - 1, -1)
    Lz = np.diag(m_vals)
    Lp = np.zeros((n, n), dtype=np.complex128)
    for k in range(n - 1):
        mv = m_vals[k + 1]
        Lp[k, k + 1] = np.sqrt(j * (j + 1) - mv * (mv + 1))
    Lx = 0.5 * (Lp + Lp.conj().T)
    Ly = -0.5j * (Lp - Lp.conj().T)
    return torch.tensor(np.stack([Lx, Ly, Lz]), dtype=torch.complex128, device=device)


@torch.no_grad()
def analyze_matrix_geometry(Y: torch.Tensor) -> Dict[str, object]:
    """Computes gauge-invariant Tr(Y^2), R^2 spectrum, and best-fit su(2) algebra error (pure diagnostic)."""
    N = Y.shape[1]
    tr_Y2 = float(torch.real(torch.sum(Y * Y.conj())))
    R2_mat = sum(Y[I] @ Y[I] for I in range(3))
    R2_mat = 0.5 * (R2_mat + R2_mat.conj().T)
    r2_evals = torch.real(safe_eigvalsh(R2_mat)).cpu().numpy()

    eps3 = [
        (0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0),
        (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0),
    ]
    num_a, den_a = 0.0, 0.0
    for I, J, K, sgn in eps3:
        C_IJ = -1j * (Y[I] @ Y[J] - Y[J] @ Y[I])
        num_a += float(torch.real(torch.trace(C_IJ @ (sgn * Y[K]))))
        den_a += float(torch.real(torch.trace(Y[K] @ Y[K])))
    alpha_fit = num_a / max(den_a, 1e-12)

    alg_res = 0.0
    for I, J, K, sgn in eps3:
        C_IJ = -1j * (Y[I] @ Y[J] - Y[J] @ Y[I])
        diff = C_IJ - alpha_fit * sgn * Y[K]
        alg_res += float(torch.real(torch.sum(diff * diff.conj())))

    rel_alg_err = float(np.sqrt(alg_res / max(2.0 * (alpha_fit ** 2) * tr_Y2, 1e-12))) if tr_Y2 > 1e-4 else 1.0
    R_rms = float(np.sqrt(max(tr_Y2 / N, 1e-12)))

    return {
        "tr_Y2": tr_Y2,
        "R_rms": R_rms,
        "r2_evals": r2_evals.tolist(),
        "r2_spread_rel": float((np.max(r2_evals) - np.min(r2_evals)) / max(np.mean(r2_evals), 1e-12)) if tr_Y2 > 1e-4 else 0.0,
        "alpha_fit": float(alpha_fit),
        "rel_alg_err": rel_alg_err,
    }


class GenesisQuantumCore:
    """
    Exact HFB Gaussian Matrix Wave-Packet + Exact 2-Flavor BdG Link-Fermion
    + Exact U(N)-Singlet Gauss-Law Casimir + Exact Bosonic Symplectic Entropy
    + 100% Gauge-Invariant Point-Probe Dirac Geometry.
    """

    def __init__(self, N: int, D: int = 3, guard: Optional[HardwareSafetyGuard] = None):
        self.guard = guard or HardwareSafetyGuard()
        self.preflight_info = self.guard.preflight_check(N=N, D=D, mode="exact_hfb")
        self.device = self.guard.device
        self.N = N
        self.D = D
        self.N2 = N * N
        self.nB = D * self.N2

        self.rdtype = torch.float64
        self.cdtype = torch.complex128

        self.T_basis = build_hermitian_basis(N, device=self.device)

        comm_T = (
            torch.einsum("aij,bjk->abik", self.T_basis, self.T_basis)
            - torch.einsum("bij,ajk->abik", self.T_basis, self.T_basis)
        )
        self.f_abc = torch.real(-1j * torch.einsum("abij,cji->abc", comm_T, self.T_basis)).contiguous()
        del comm_T

        self.f_flat = self.f_abc.reshape(self.N2, self.N2 * self.N2).contiguous()
        self.f_AC_p = self.f_abc.permute(0, 2, 1).reshape(self.N2 * self.N2, self.N2).contiguous()
        self.f_Cp_B = self.f_abc.permute(2, 1, 0).reshape(self.N2 * self.N2, self.N2).contiguous()

        self.eps3_pairs = [
            (0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0),
            (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0),
        ]

        self.sigma = torch.stack([
            torch.tensor([[0, 1], [1, 0]], dtype=self.cdtype),
            torch.tensor([[0, -1j], [1j, 0]], dtype=self.cdtype),
            torch.tensor([[1, 0], [0, -1]], dtype=self.cdtype),
        ]).to(self.device)

        self.eye_N = torch.eye(N, dtype=self.cdtype, device=self.device)
        self.eye_N2 = torch.eye(self.N2, dtype=self.rdtype, device=self.device)
        self.eye_nB = torch.eye(self.nB, dtype=self.rdtype, device=self.device)

    @staticmethod
    def _safe_eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return safe_eigh(A)

    @staticmethod
    def _safe_eigvalsh(A: torch.Tensor) -> torch.Tensor:
        return safe_eigvalsh(A)

    @staticmethod
    def _build_hermitian_basis(N: int) -> torch.Tensor:
        return build_hermitian_basis(N)

    def mats_to_vec(self, Y: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.einsum("ijk,akj->ia", Y, self.T_basis)).reshape(-1)

    def vec_to_mats(self, y_vec: torch.Tensor) -> torch.Tensor:
        y_c = y_vec.reshape(self.D, self.N2).to(self.cdtype)
        return torch.einsum("ia,ajk->ijk", y_c, self.T_basis)

    def project_traceless(self, y_vec: torch.Tensor) -> torch.Tensor:
        y = y_vec.reshape(self.D, self.N2).clone()
        tr = torch.sum(y[:, :self.N], dim=1, keepdim=True) / float(self.N)
        y[:, :self.N] -= tr
        return y.reshape(-1)

    # --------------------------------------------------------------------------
    # 1. Exact Self-Consistent HFB Effective Hessian & Quantum MQM Energy
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def build_h2_and_kloop(
        self, y_vec: torch.Tensor, Sigma_x: torch.Tensor,
        g: float = 0.25, mu: float = 1.0, nu: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y_vec.reshape(self.D, self.N2)
        Sx = Sigma_x.reshape(self.D, self.N2, self.D, self.N2)

        F_mat = (y @ self.f_flat).reshape(self.D, self.N2, self.N2)
        C_IJ = torch.einsum("ibc,jb->ijc", F_mat, y)

        H2 = torch.zeros((self.D, self.N2, self.D, self.N2), dtype=self.rdtype, device=self.device)
        K_loop = torch.zeros_like(H2)
        g2 = g * g

        for I in range(self.D):
            H2[I, :, I, :] += (mu * mu) * self.eye_N2
            for J in range(self.D):
                if I == J:
                    continue
                H2[I, :, I, :] += g2 * (F_mat[J] @ F_mat[J].T)
                c_f = torch.tensordot(C_IJ[I, J], self.f_abc, dims=([0], [2]))
                H2[I, :, J, :] += g2 * (-(F_mat[J] @ F_mat[I].T) + c_f)

                M_h = (self.f_AC_p @ Sx[J, :, J, :]).reshape(self.N2, self.N2 * self.N2)
                K_loop[I, :, I, :] += g2 * (M_h @ self.f_Cp_B)

                t_C = torch.tensordot(Sx[I, :, J, :], self.f_abc, dims=([0, 1], [0, 1]))
                K_loop[I, :, J, :] += g2 * torch.tensordot(t_C, self.f_abc, dims=([0], [2]))

                M_f = (self.f_AC_p @ Sx[J, :, I, :]).reshape(self.N2, self.N2 * self.N2)
                K_loop[I, :, J, :] += -g2 * (M_f @ self.f_Cp_B)

        if self.D == 3 and nu != 0.0:
            for I, J, K, sgn in self.eps3_pairs:
                H2[I, :, J, :] += (-2.0 * nu * g * sgn) * F_mat[K]

        H2 = H2.reshape(self.nB, self.nB)
        K_loop = K_loop.reshape(self.nB, self.nB)
        return 0.5 * (H2 + H2.T), 0.5 * (K_loop + K_loop.T)

    @torch.no_grad()
    def solve_hfb_omega(
        self, y_vec: torch.Tensor, m: float = 1.0, g: float = 0.25, mu: float = 1.0, nu: float = 3.0,
        max_iter: int = 12, tol: float = 1e-6, om_min: float = 0.15,
        Omega_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Omega = (mu + 0.8) * self.eye_nB.clone() if Omega_init is None else Omega_init.clone()
        om_min_sq = om_min * om_min

        for _ in range(max_iter):
            ew, ev = safe_eigh(0.5 * (Omega + Omega.T))
            ew = torch.clamp(ew, min=om_min)
            Sx = (ev * (0.5 / ew)) @ ev.T
            H2, K_loop = self.build_h2_and_kloop(y_vec, Sx, g=g, mu=mu, nu=nu)
            ek, uk = safe_eigh(H2 + K_loop)
            om_new = torch.sqrt(torch.clamp(m * ek, min=om_min_sq))
            Omega_new = (uk * om_new) @ uk.T
            diff = torch.linalg.norm(Omega_new - Omega) / torch.clamp(torch.linalg.norm(Omega), min=1e-12)
            Omega = 0.65 * Omega_new + 0.35 * Omega
            if float(diff) < tol:
                break

        ew, ev = safe_eigh(0.5 * (Omega + Omega.T))
        ew = torch.clamp(ew, min=om_min)
        Sx = (ev * (0.5 / ew)) @ ev.T
        return Omega, Sx, ew, ev

    @torch.no_grad()
    def mqm_quantum_energy(
        self, y_vec: torch.Tensor, Omega: torch.Tensor, Sx: torch.Tensor,
        m: float = 1.0, g: float = 0.25, mu: float = 1.0, nu: float = 3.0
    ) -> Dict[str, float]:
        y = y_vec.reshape(self.D, self.N2)
        F_mat = (y @ self.f_flat).reshape(self.D, self.N2, self.N2)
        C_IJ = torch.einsum("ibc,jb->ijc", F_mat, y)

        ew = safe_eigvalsh(0.5 * (Omega + Omega.T))
        E_kin = float(0.25 / m * torch.sum(ew))
        E_mass = float(0.5 * (mu * mu) * (torch.sum(y * y) + torch.trace(Sx)))
        E_cl_ym = float(0.25 * (g * g) * torch.sum(C_IJ * C_IJ))
        E_cl_flux = 0.0
        if self.D == 3 and nu != 0.0:
            for I, J, K, sgn in self.eps3_pairs:
                E_cl_flux += -(nu * g / 3.0) * sgn * float(torch.dot(y[I], C_IJ[J, K]))

        H2_pot, K_loop = self.build_h2_and_kloop(y_vec, Sx, g=g, mu=mu, nu=nu)
        E_fluc2 = float(0.5 * torch.sum((H2_pot - (mu * mu) * self.eye_nB) * Sx))
        E_ym4 = float(0.25 * torch.sum(K_loop * Sx))

        return {
            "E_kin": E_kin,
            "E_mass": E_mass,
            "E_cl": E_cl_ym + E_cl_flux,
            "E_fluc2": E_fluc2,
            "E_ym4": E_ym4,
            "E_mqm": E_kin + E_mass + E_cl_ym + E_cl_flux + E_fluc2 + E_ym4,
        }

    @torch.no_grad()
    def mqm_y_energy_fixed_sx(
        self, y_vec: torch.Tensor, Sx: torch.Tensor,
        g: float = 0.25, mu: float = 1.0, nu: float = 3.0
    ) -> float:
        """Exact y-dependent MQM energy at fixed Sx (skips redundant K_loop(Sx) recompute)."""
        y = y_vec.reshape(self.D, self.N2)
        Sx_4 = Sx.reshape(self.D, self.N2, self.D, self.N2)
        F_mat = (y @ self.f_flat).reshape(self.D, self.N2, self.N2)
        C_IJ = torch.einsum("ibc,jb->ijc", F_mat, y)
        g2v = g * g
        e = 0.5 * (mu * mu) * float(torch.sum(y * y)) + 0.25 * g2v * float(torch.sum(C_IJ * C_IJ))
        if self.D == 3 and nu != 0.0:
            for I, J, K, sgn in self.eps3_pairs:
                e += -(nu * g / 3.0) * sgn * float(torch.dot(y[I], C_IJ[J, K]))
                e += -(nu * g * sgn) * float(torch.sum(F_mat[K] * Sx_4[I, :, J, :]))
        for I in range(self.D):
            for J in range(self.D):
                if I == J:
                    continue
                e += 0.5 * g2v * float(torch.sum((F_mat[J] @ F_mat[J].T) * Sx_4[I, :, I, :]))
                e += -0.5 * g2v * float(torch.sum((F_mat[J] @ F_mat[I].T) * Sx_4[I, :, J, :]))
                t_C = torch.tensordot(Sx_4[I, :, J, :], self.f_abc, dims=([0, 1], [0, 1]))
                e += 0.5 * g2v * float(torch.dot(C_IJ[I, J], t_C))
        return e

    # --------------------------------------------------------------------------
    # 2. Exact Noncommutative Laplacian & 2-Flavor BdG Link-Fermion Sector
    # --------------------------------------------------------------------------
    def build_laplacian_matrix(self, Y: torch.Tensor) -> torch.Tensor:
        Delta = torch.zeros((self.N2, self.N2), dtype=self.rdtype, device=self.device)
        for I in range(self.D):
            comm = Y[I].unsqueeze(0) @ self.T_basis - self.T_basis @ Y[I].unsqueeze(0)
            Delta += torch.real(torch.einsum("ajk,bjk->ab", comm.conj(), comm))
        return 0.5 * (Delta + Delta.T)

    def bdg_link_sector(
        self, Y: torch.Tensor, kappa: float = 0.15, Nf: int = 1, g1: float = 0.015, g2: float = 0.0001
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Delta = self.build_laplacian_matrix(Y)
        lam, U_lap = safe_eigh(Delta)
        lam = torch.clamp(lam, min=0.0)
        f = g1 * lam + g2 * (lam * lam)
        E_tn = kappa * Nf * torch.sum(f - torch.sqrt(f * f + 1.0))
        E_mats = torch.einsum("ak,ajl->kjl", U_lap.to(self.cdtype), self.T_basis)
        return E_tn, lam, f, E_mats, U_lap

    def build_traceless_basis(self) -> torch.Tensor:
        """Exact orthonormal (N^2 - 1) basis of traceless Hermitian su(N) matrices orthogonal to I_N/sqrt(N)."""
        basis = []
        # Diagonal Cartan generators (N - 1)
        for m in range(1, self.N):
            T = torch.zeros((self.N, self.N), dtype=self.cdtype, device=self.device)
            val = 1.0 / np.sqrt(float(m * (m + 1)))
            for i in range(m):
                T[i, i] = val
            T[m, m] = -float(m) * val
            basis.append(T)
        inv_s2 = 1.0 / np.sqrt(2.0)
        for a in range(self.N):
            for b in range(a + 1, self.N):
                T = torch.zeros((self.N, self.N), dtype=self.cdtype, device=self.device)
                T[a, b] = inv_s2
                T[b, a] = inv_s2
                basis.append(T)
        for a in range(self.N):
            for b in range(a + 1, self.N):
                T = torch.zeros((self.N, self.N), dtype=self.cdtype, device=self.device)
                T[a, b] = 1j * inv_s2
                T[b, a] = -1j * inv_s2
                basis.append(T)
        return torch.stack(basis, dim=0)

    def bdg_link_sector_traceless(
        self, Y: torch.Tensor, kappa: float = 0.15, Nf: int = 1,
        c1: float = 1.0, c2: float = 0.0, normalize_by_R2: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Traceless su(N) 2-flavor BdG link sector strictly projecting out the geometry-blind
        U(1) identity mode E_0 = I_N / sqrt(N) BEFORE diagonalization (safe on degenerate spectra).
        When normalize_by_R2=True, Y is rescaled by U(N)-invariant R_rms = sqrt(Tr(Y^2)/N)
        so Delta_{Y/R_rms} has O(1) spectrum independent of N.
        """
        tr_y2 = torch.real(torch.sum(Y * Y.conj()))
        if float(tr_y2) < 1e-20:
            T_su = self.build_traceless_basis()
            zeros = torch.zeros(self.N2 - 1, dtype=self.rdtype, device=self.device)
            return torch.zeros((), dtype=self.rdtype, device=self.device), zeros, zeros, T_su

        if normalize_by_R2:
            R_rms = torch.sqrt(tr_y2 / float(self.N))
            Y_use = Y / R_rms
        else:
            Y_use = Y

        T_su = self.build_traceless_basis()
        Delta_su = torch.zeros((self.N2 - 1, self.N2 - 1), dtype=self.rdtype, device=self.device)
        for I in range(self.D):
            comm = Y_use[I].unsqueeze(0) @ T_su - T_su @ Y_use[I].unsqueeze(0)
            Delta_su += torch.real(torch.einsum("ajk,bjk->ab", comm.conj(), comm))
        Delta_su = 0.5 * (Delta_su + Delta_su.T)

        lam_su, U_su = safe_eigh(Delta_su)
        lam_su = torch.clamp(lam_su, min=0.0)
        f_su = c1 * lam_su + c2 * (lam_su * lam_su)
        E_tn_su = kappa * Nf * torch.sum(f_su - torch.sqrt(f_su * f_su + 1.0))
        E_mats_su = torch.einsum("ak,ajl->kjl", U_su.to(self.cdtype), T_su)
        return E_tn_su, lam_su, f_su, E_mats_su

    def bdg_subregion_entropy_traceless(
        self, E_mats_su: torch.Tensor, f_su: torch.Tensor, Q_in: torch.Tensor, Nf: int = 1
    ) -> torch.Tensor:
        """
        Exact Peschel von Neumann entropy of the traceless su(N) BdG link sector
        with the U(1) identity mode E_0 = I_N/sqrt(N) strictly fixed in the unentangled vacuum
        (n_0 = 0, m_0 = 1, a_0 = 0).
        """
        M = Q_in.shape[1]
        eps = torch.sqrt(f_su * f_su + 1.0)
        n_k = ((eps - f_su) / (2.0 * eps)).to(self.cdtype)
        m_k = ((eps + f_su) / (2.0 * eps)).to(self.cdtype)
        a_k = (-1j / (2.0 * eps)).to(self.cdtype)

        B_su = torch.einsum("ip,kij,jq->kpq", Q_in.conj(), E_mats_su, Q_in).reshape(self.N2 - 1, M * M)
        B_su_H = B_su.conj().T

        # Excluded U(1) identity mode E_0 = I_N / sqrt(N) has B_0,(pq) = delta_pq / sqrt(N)
        eye_M = torch.eye(M, dtype=self.cdtype, device=self.device).reshape(1, M * M) / np.sqrt(float(self.N))
        P0 = eye_M.conj().T @ eye_M  # m_0 = 1 contribution to G22

        G11 = (B_su_H * n_k) @ B_su
        G22 = P0 + (B_su_H * m_k) @ B_su
        G21 = (B_su_H * a_k) @ B_su
        G12 = G21.conj().T

        G_sub = torch.cat([torch.cat([G11, G12], dim=1), torch.cat([G21, G22], dim=1)], dim=0)
        G_sub = 0.5 * (G_sub + G_sub.conj().T)

        ev = torch.clamp(safe_eigvalsh(G_sub), min=1e-13, max=1.0 - 1e-13)
        return -float(Nf) * torch.sum(ev * torch.log(ev) + (1.0 - ev) * torch.log(1.0 - ev))

    @torch.no_grad()
    def exact_grad_y(
        self, y_vec: torch.Tensor, Sx: torch.Tensor,
        g: float = 0.25, mu: float = 1.0, nu: float = 3.0,
        kappa: float = 0.15, Nf: int = 1, g1: float = 0.015, g2: float = 0.0001
    ) -> Tuple[torch.Tensor, float]:
        y = y_vec.reshape(self.D, self.N2)
        Sx_4 = Sx.reshape(self.D, self.N2, self.D, self.N2)
        Y = self.vec_to_mats(y_vec)

        F_mat = (y @ self.f_flat).reshape(self.D, self.N2, self.N2)
        C_IJ = torch.einsum("ibc,jb->ijc", F_mat, y)
        grad_y = (mu * mu) * y.clone()
        g2_val = g * g

        for I in range(self.D):
            for J in range(self.D):
                if I == J:
                    continue
                grad_y[I] += -g2_val * (F_mat[J] @ C_IJ[I, J])
                grad_y[I] += g2_val * (self.f_flat @ (Sx_4[J, :, J, :] @ F_mat[I]).reshape(-1))
                grad_y[I] += -g2_val * (self.f_flat @ (Sx_4[I, :, J, :].T @ F_mat[J]).reshape(-1))
                t_C = torch.tensordot(Sx_4[I, :, J, :], self.f_abc, dims=([0, 1], [0, 1]))
                grad_y[I] += -g2_val * (F_mat[J] @ t_C)

        if self.D == 3 and nu != 0.0:
            for I, J, K, sgn in self.eps3_pairs:
                grad_y[I] += -(nu * g * sgn) * C_IJ[J, K]
                grad_y[I] += -(nu * g * sgn) * (self.f_flat @ Sx_4[J, :, K, :].reshape(-1))

        E_tn, lam, f, _, U_lap = self.bdg_link_sector(Y, kappa=kappa, Nf=Nf, g1=g1, g2=g2)
        dE_dlam = kappa * Nf * (1.0 - f / torch.sqrt(f * f + 1.0)) * (g1 + 2.0 * g2 * lam)
        W_lap = (U_lap * dE_dlam) @ U_lap.T
        for I in range(self.D):
            grad_y[I] += 2.0 * (self.f_flat @ (W_lap @ F_mat[I]).reshape(-1))

        return grad_y.reshape(-1), float(E_tn)

    # --------------------------------------------------------------------------
    # 3. Differentiable Subregion Entanglement (Peschel BdG + U(N) Casimir + S_bos)
    # --------------------------------------------------------------------------
    def bdg_subregion_entropy(
        self, E_mats: torch.Tensor, f: torch.Tensor, Q_in: torch.Tensor, Nf: int = 1
    ) -> torch.Tensor:
        M = Q_in.shape[1]
        eps = torch.sqrt(f * f + 1.0)
        n_k = ((eps - f) / (2.0 * eps)).to(self.cdtype)
        m_k = ((eps + f) / (2.0 * eps)).to(self.cdtype)
        a_k = (-1j / (2.0 * eps)).to(self.cdtype)

        B = torch.einsum("ip,kij,jq->kpq", Q_in.conj(), E_mats, Q_in).reshape(self.N2, M * M)
        BH = B.conj().T
        G11 = (BH * n_k) @ B
        G22 = (BH * m_k) @ B
        G21 = (BH * a_k) @ B
        G12 = G21.conj().T

        G_sub = torch.cat([torch.cat([G11, G12], dim=1), torch.cat([G21, G22], dim=1)], dim=0)
        G_sub = 0.5 * (G_sub + G_sub.conj().T)

        ev = torch.clamp(safe_eigvalsh(G_sub), min=1e-13, max=1.0 - 1e-13)
        return -float(Nf) * torch.sum(ev * torch.log(ev) + (1.0 - ev) * torch.log(1.0 - ev))

    def precompute_boson_entanglement_cache(
        self, Omega: torch.Tensor, Y: Optional[torch.Tensor] = None, project_gauss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Precomputes Omega modes and optional exact U(N)-singlet Coulomb-transverse projection P_phys."""
        Om_sym = 0.5 * (Omega + Omega.T)
        omegas, evecs = safe_eigh(Om_sym)
        omegas = torch.clamp(omegas, min=1e-12)
        V_modes = torch.einsum(
            "ria,ajk->rijk",
            evecs.T.reshape(self.nB, self.D, self.N2).to(self.cdtype),
            self.T_basis,
        )
        Sx = (evecs * (0.5 / omegas)) @ evecs.T
        Sp = (evecs * (0.5 * omegas)) @ evecs.T

        if Y is not None and project_gauss:
            y_vec = self.mats_to_vec(Y).reshape(self.D, self.N2)
            F_Y = (y_vec @ self.f_flat).reshape(self.D, self.N2, self.N2)
            G_basis = F_Y.permute(1, 0, 2).reshape(self.N2, self.nB).T
            U_g, s_g, _ = torch.linalg.svd(G_basis, full_matrices=False)
            r_g = int(torch.sum(s_g > 1e-7 * torch.clamp(torch.max(s_g), min=1.0))) if torch.max(s_g) > 1e-10 else 0
            P_phys = (self.eye_nB - U_g[:, :r_g] @ U_g[:, :r_g].T) if r_g > 0 else self.eye_nB
            Om_phys = P_phys @ Om_sym @ P_phys
            Om_phys = 0.5 * (Om_phys + Om_phys.T)
            om_p, ev_p = safe_eigh(Om_phys)
            om_p = torch.where(om_p > 1e-9, om_p, torch.zeros_like(om_p))
            V_phys = torch.einsum(
                "ria,ajk->rijk",
                ev_p.T.reshape(self.nB, self.D, self.N2).to(self.cdtype),
                self.T_basis,
            )
        else:
            om_p, V_phys = omegas, V_modes

        return {
            "omegas": omegas,
            "evecs": evecs,
            "V_modes": V_modes,
            "omegas_phys": om_p,
            "V_modes_phys": V_phys,
            "Sx": Sx,
            "Sp": Sp,
        }

    def gauss_casimir_and_boson_entropy(
        self, Y: torch.Tensor, cache: Dict[str, torch.Tensor], Q_in: torch.Tensor,
        T_M: Optional[torch.Tensor] = None, full_casimir: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M = Q_in.shape[1]
        M2 = M * M
        if T_M is None:
            T_M = build_hermitian_basis(M, device=self.device)

        omegas = cache["omegas"]
        V_modes = cache["V_modes"]
        omegas_p = cache.get("omegas_phys", omegas)
        V_modes_p = cache.get("V_modes_phys", V_modes)
        Sx = cache["Sx"]
        Sp = cache["Sp"]

        Y_in = torch.einsum("ip,dij,jq->dpq", Q_in.conj(), Y, Q_in)
        V_in_p = torch.einsum("ip,rdij,jq->rdpq", Q_in.conj(), V_modes_p, Q_in)

        comm_YV = (
            torch.einsum("dab,rdbc->rac", Y_in, V_in_p)
            - torch.einsum("rdab,dbc->rac", V_in_p, Y_in)
        )
        term1 = 0.5 * torch.sum(omegas_p * torch.real(torch.einsum("rab,rab->r", comm_YV.conj(), comm_YV)))

        term2 = torch.zeros((), dtype=self.rdtype, device=self.device)
        if full_casimir:
            V_in = torch.einsum("ip,rdij,jq->rdpq", Q_in.conj(), V_modes, Q_in)
            sw = torch.sqrt(omegas)
            rd2 = (sw.unsqueeze(0) / sw.unsqueeze(1) - sw.unsqueeze(1) / sw.unsqueeze(0)) ** 2
            chunk_r = 64
            for r0 in range(0, self.nB, chunk_r):
                r1 = min(r0 + chunk_r, self.nB)
                Vc = V_in[r0:r1]
                comm_VV = (
                    torch.einsum("rdab,sdbc->rsac", Vc, V_in)
                    - torch.einsum("sdab,rdbc->rsac", V_in, Vc)
                )
                norm2_VV = torch.real(torch.einsum("rsac,rsac->rs", comm_VV.conj(), comm_VV))
                term2 = term2 + 0.125 * torch.sum(rd2[r0:r1] * norm2_VV)

        casimir = term1 + term2

        # Exact Bosonic Symplectic Entropy via Cholesky similarity L_x^T Sp_in L_x ~ Sp_in Sx_in
        T_N_in = torch.einsum("ip,aij,jq->apq", Q_in.conj(), self.T_basis, Q_in)
        P_blk = torch.real(torch.einsum("bqp,apq->ba", T_M, T_N_in))

        Sx_4 = Sx.reshape(self.D, self.N2, self.D, self.N2)
        Sp_4 = Sp.reshape(self.D, self.N2, self.D, self.N2)
        Sx_in = torch.einsum("pa,iajb,qb->ipjq", P_blk, Sx_4, P_blk).reshape(self.D * M2, self.D * M2)
        Sp_in = torch.einsum("pa,iajb,qb->ipjq", P_blk, Sp_4, P_blk).reshape(self.D * M2, self.D * M2)

        Sx_in_sym = 0.5 * (Sx_in + Sx_in.T)
        Sp_in_sym = 0.5 * (Sp_in + Sp_in.T)
        eye_in = torch.eye(self.D * M2, dtype=self.rdtype, device=self.device)
        L_x = torch.linalg.cholesky(Sx_in_sym + 1e-14 * eye_in)
        symp_mat = L_x.T @ Sp_in_sym @ L_x
        symp_sq = safe_eigvalsh(0.5 * (symp_mat + symp_mat.T))
        nu_vals = torch.sqrt(torch.clamp(symp_sq, min=0.25 + 1e-12))

        mask = nu_vals > (0.5 + 1e-10)
        if torch.any(mask):
            nv = nu_vals[mask]
            S_bos = torch.sum((nv + 0.5) * torch.log(nv + 0.5) - (nv - 0.5) * torch.log(nv - 0.5))
        else:
            S_bos = torch.zeros((), dtype=self.rdtype, device=self.device)

        return casimir, S_bos

    # --------------------------------------------------------------------------
    # 4. 100% U(N) Gauge-Invariant Point-Probe Dirac Geometry & Quantum Metric
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def point_probe_dirac(self, Y: torch.Tensor, y_pt: torch.Tensor) -> Tuple[float, torch.Tensor]:
        D_mat = torch.zeros((2 * self.N, 2 * self.N), dtype=self.cdtype, device=self.device)
        for I in range(3):
            D_mat += torch.kron((Y[I] - y_pt[I] * self.eye_N).contiguous(), self.sigma[I].contiguous())
        D_mat = 0.5 * (D_mat + D_mat.conj().T)
        evals, evecs = safe_eigh(D_mat)
        idx_min = int(torch.argmin(torch.abs(evals)))
        return float(torch.abs(evals[idx_min])), evecs[:, idx_min]

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
        if not torch.any(mask):
            return np.zeros((3, 3), dtype=float), 0.0
        gap = float(torch.min(torch.abs(diffs[mask])))

        psi_other = evecs[:, mask]
        inv_d = (1.0 / diffs[mask]).to(self.cdtype)
        M_mat = torch.zeros((3, psi_other.shape[1]), dtype=self.cdtype, device=self.device)
        for a in range(3):
            v_a = torch.kron(self.eye_N, self.sigma[a].contiguous()) @ psi0
            M_mat[a] = (psi_other.conj().T @ v_a) * inv_d

        g_mat = torch.real(M_mat.conj() @ M_mat.T).cpu().numpy()
        return 0.5 * (g_mat + g_mat.T), gap
