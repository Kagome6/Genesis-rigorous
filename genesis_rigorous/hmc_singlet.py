"""
Exact U(N)-Singlet Gauged Lattice Hybrid Monte Carlo (HMC) for (0+1)D Bosonic
Yang-Mills-Myers Matrix Quantum Mechanics.

Evaluates the exact finite-temperature U(N)-singlet Euclidean path integral:
    Z_singlet(beta) = int dU DX exp(-S_lat[X, U]),
with L Euclidean time slices (spacing a = beta / L), full N x N Hermitian
matrices X_I(t), and a single dynamical unitary Polyakov-loop holonomy U in U(N)
closing the thermal circle t = L-1 -> 0 (exact gauge-reduced form of L links,
with zero sign problem and exact U(N) singlet projection satisfying Elitzur's theorem).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch


EPS3_PAIRS = (
    (0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0),
    (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0),
)


class ExactSingletMQMHMC:
    """
    Batched CUDA/CPU Hybrid Monte Carlo sampler for (0+1)D gauged bosonic MQM.
    Runs B independent Markov chains in parallel in double precision (complex128).
    """

    def __init__(
        self,
        N: int,
        L: int = 16,
        beta: float = 4.0,
        m: float = 1.0,
        g: float = 0.25,
        mu: float = 1.0,
        nu: float = 3.0,
        gauged: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.N = int(N)
        self.L = int(L)
        self.beta = float(beta)
        self.a = float(beta) / float(L)
        self.m = float(m)
        self.g = float(g)
        self.mu = float(mu)
        self.nu = float(nu)
        self.gauged = bool(gauged)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eye_N = torch.eye(self.N, dtype=torch.complex128, device=self.device)

    @staticmethod
    def _hermitianize(M: torch.Tensor) -> torch.Tensor:
        return 0.5 * (M + M.conj().transpose(-2, -1))

    @staticmethod
    def _reunitarize(U: torch.Tensor) -> torch.Tensor:
        # Polar projection U -> U (U^H U)^(-1/2) via SVD to eliminate floating-point drift
        u_svd, _, vh_svd = torch.linalg.svd(U)
        return u_svd @ vh_svd

    def sample_hermitian_momentum(self, shape_prefix: Tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
        """
        Samples Hermitian P with exact orthonormal Gaussian measure exp(-1/2 Tr(P^2)):
        P_aa ~ N(0, 1), Re(P_ab), Im(P_ab) ~ N(0, 1/2) for a < b, so E[1/2 Tr(P^2)] = N^2 / 2.
        """
        full_shape = shape_prefix + (self.N, self.N)
        re = torch.randn(full_shape, generator=gen, dtype=torch.float64)
        im = torch.randn(full_shape, generator=gen, dtype=torch.float64)
        Z = (re + 1j * im).to(self.device)
        return 0.5 * (Z + Z.conj().transpose(-2, -1))

    @torch.no_grad()
    def action_and_observables(self, X: torch.Tensor, U: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes per-chain action components S_kin, S_mass, S_ym, S_flux, total S_lat,
        and per-slice gauge-invariant observables.
        X: (B, L, 3, N, N), U: (B, N, N)
        """
        a = self.a
        m = self.m
        g = self.g
        mu = self.mu
        nu = self.nu

        # Kinetic links: interior t=0..L-2 and twisted closing link t=L-1 -> 0
        X_next = torch.empty_like(X)
        X_next[:, :-1] = X[:, 1:]
        # Closing link: U X(0) U^H
        U_exp = U.unsqueeze(1)  # (B, 1, N, N)
        X_next[:, -1] = U_exp @ X[:, 0] @ U_exp.conj().transpose(-2, -1)

        dX = X_next - X
        S_kin = (0.5 * m / a) * torch.real(torch.sum(dX * dX.conj(), dim=(1, 2, 3, 4)))

        # Potential summed over time slices t=0..L-1 (loop-free 3-commutator reuse)
        X0, X1, X2 = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        tr_X2_t = torch.real(torch.sum(X * X.conj(), dim=(2, 3, 4)))  # (B, L)
        V_mass_t = 0.5 * (mu * mu) * tr_X2_t

        C01 = X0 @ X1 - X1 @ X0
        C12 = X1 @ X2 - X2 @ X1
        C20 = X2 @ X0 - X0 @ X2
        V_ym_t = (0.5 * g * g) * torch.real(
            torch.sum(C01 * C01.conj() + C12 * C12.conj() + C20 * C20.conj(), dim=(-2, -1))
        )

        if nu != 0.0:
            # sum_{eps} eps_{IJK} Tr(X_I [X_J, X_K]) = 2 * Tr(X0 C12 + X1 C20 + X2 C01)
            flux_tr = (
                torch.einsum("btij,btji->bt", X0, C12)
                + torch.einsum("btij,btji->bt", X1, C20)
                + torch.einsum("btij,btji->bt", X2, C01)
            )
            V_flux_t = torch.real((2j * nu * g / 3.0) * flux_tr)
        else:
            V_flux_t = torch.zeros_like(tr_X2_t)

        S_mass = a * torch.sum(V_mass_t, dim=1)
        S_ym = a * torch.sum(V_ym_t, dim=1)
        S_flux = a * torch.sum(V_flux_t, dim=1)
        S_tot = S_kin + S_mass + S_ym + S_flux

        # Time-averaged observables per chain
        mean_trX2 = torch.mean(tr_X2_t, dim=1)
        mean_V_mass = torch.mean(V_mass_t, dim=1)
        mean_V_ym = torch.mean(V_ym_t, dim=1)
        mean_V_flux = torch.mean(V_flux_t, dim=1)
        # Exact quantum virial energy estimator: <H> = 2<V_mass> + (5/2)<V_flux> + 3<V_ym>
        E_virial = 2.0 * mean_V_mass + 2.5 * mean_V_flux + 3.0 * mean_V_ym
        # Discrete Schwinger-Dyson identity lhs (exact expectation == D * N^2 * L)
        sd_lhs = 2.0 * S_kin + 2.0 * S_mass + 3.0 * S_flux + 4.0 * S_ym
        polyakov_abs = torch.abs(torch.einsum("bii->b", U)) / float(self.N)

        return {
            "S_tot": S_tot,
            "S_kin": S_kin,
            "S_mass": S_mass,
            "S_ym": S_ym,
            "S_flux": S_flux,
            "mean_trX2": mean_trX2,
            "mean_V_mass": mean_V_mass,
            "mean_V_ym": mean_V_ym,
            "mean_V_flux": mean_V_flux,
            "E_virial": E_virial,
            "sd_lhs": sd_lhs,
            "polyakov_abs": polyakov_abs,
        }

    @torch.no_grad()
    def forces(self, X: torch.Tensor, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact analytical Hermitian gradients F_X = dS/dX and F_U = dS/d(i eps)
        satisfying dS = Tr(F_X dX) + Tr(F_U eps) for delta U = i eps U.
        """
        a = self.a
        m = self.m
        g = self.g
        mu = self.mu
        nu = self.nu
        g2 = g * g

        # Kinetic force on X
        U_exp = U.unsqueeze(1)  # (B, 1, N, N)
        U_dag = U_exp.conj().transpose(-2, -1)

        X_prev = torch.empty_like(X)
        X_prev[:, 1:] = X[:, :-1]
        X_prev[:, 0] = U_dag @ X[:, -1] @ U_exp

        X_next = torch.empty_like(X)
        X_next[:, :-1] = X[:, 1:]
        X_next[:, -1] = U_exp @ X[:, 0] @ U_dag

        F_X = (m / a) * (2.0 * X - X_prev - X_next)

        # Potential force on X: a * dV/dX (loop-free 3-commutator reuse)
        X0, X1, X2 = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        C01 = X0 @ X1 - X1 @ X0
        C12 = X1 @ X2 - X2 @ X1
        C20 = X2 @ X0 - X0 @ X2
        ag2 = a * g2
        F0 = (a * mu * mu) * X0 - ag2 * ((X1 @ C01 - C01 @ X1) - (X2 @ C20 - C20 @ X2))
        F1 = (a * mu * mu) * X1 + ag2 * ((X0 @ C01 - C01 @ X0) - (X2 @ C12 - C12 @ X2))
        F2 = (a * mu * mu) * X2 + ag2 * ((X1 @ C12 - C12 @ X1) - (X0 @ C20 - C20 @ X0))
        if nu != 0.0:
            c_fl = 2j * a * nu * g
            F0 = F0 + c_fl * C12
            F1 = F1 + c_fl * C20
            F2 = F2 + c_fl * C01
        F_X = F_X + torch.stack([F0, F1, F2], dim=2)
        F_X = self._hermitianize(F_X)

        # Unitary force on Polyakov holonomy U: F_U = -(i m / a) sum_I [U X_I(0) U^H, X_I(L-1)]
        if self.gauged:
            UX0U = U_exp @ X[:, 0] @ U_dag  # (B, 3, N, N)
            X_last = X[:, -1]               # (B, 3, N, N)
            comm_sum = torch.sum(UX0U @ X_last - X_last @ UX0U, dim=1)  # (B, N, N)
            F_U = self._hermitianize((-1j * m / a) * comm_sum)
        else:
            F_U = torch.zeros_like(U)

        return F_X, F_U

    @torch.no_grad()
    def hmc_step(
        self,
        X: torch.Tensor,
        U: torch.Tensor,
        dt: float,
        n_leapfrog: int,
        gen: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single batched leapfrog HMC trajectory with exact Metropolis accept/reject."""
        B = X.shape[0]
        P_X = self.sample_hermitian_momentum((B, self.L, 3), gen)
        P_U = self.sample_hermitian_momentum((B,), gen) if self.gauged else torch.zeros_like(U)

        S0 = self.action_and_observables(X, U)["S_tot"]
        K0 = 0.5 * torch.real(torch.sum(P_X * P_X.conj(), dim=(1, 2, 3, 4)))
        if self.gauged:
            K0 = K0 + 0.5 * torch.real(torch.sum(P_U * P_U.conj(), dim=(1, 2)))
        H0 = S0 + K0

        Xn = X.clone()
        Un = U.clone()
        PXn = P_X.clone()
        PUn = P_U.clone()

        F_X, F_U = self.forces(Xn, Un)
        PXn = PXn - 0.5 * dt * F_X
        if self.gauged:
            PUn = PUn - 0.5 * dt * F_U

        for step in range(n_leapfrog):
            Xn = self._hermitianize(Xn + dt * PXn)
            if self.gauged:
                Un = torch.linalg.matrix_exp(1j * dt * PUn) @ Un
            F_X, F_U = self.forces(Xn, Un)
            if step < n_leapfrog - 1:
                PXn = PXn - dt * F_X
                if self.gauged:
                    PUn = PUn - dt * F_U
            else:
                PXn = PXn - 0.5 * dt * F_X
                if self.gauged:
                    PUn = PUn - 0.5 * dt * F_U

        if self.gauged:
            Un = self._reunitarize(Un)

        S1 = self.action_and_observables(Xn, Un)["S_tot"]
        K1 = 0.5 * torch.real(torch.sum(PXn * PXn.conj(), dim=(1, 2, 3, 4)))
        if self.gauged:
            K1 = K1 + 0.5 * torch.real(torch.sum(PUn * PUn.conj(), dim=(1, 2)))
        H1 = S1 + K1

        dH = H1 - H0
        u_rand = torch.rand(B, generator=gen, dtype=torch.float64).to(self.device)
        accept = (dH <= 0.0) | (u_rand < torch.exp(torch.clamp(-dH, min=-50.0, max=0.0)))
        # Reject NaN/Inf proposals safely
        accept = accept & torch.isfinite(dH)

        m_X = accept.view(B, 1, 1, 1, 1)
        m_U = accept.view(B, 1, 1)
        X_out = torch.where(m_X, Xn, X)
        U_out = torch.where(m_U, Un, U)
        return X_out, U_out, accept, dH

    @torch.no_grad()
    def run_chains(
        self,
        X_init: torch.Tensor,
        U_init: Optional[torch.Tensor] = None,
        n_therm: int = 200,
        n_meas: int = 400,
        meas_every: int = 2,
        dt: float = 0.03,
        n_leapfrog: int = 12,
        adapt_dt: bool = True,
        seed: int = 20260,
    ) -> Dict[str, object]:
        """
        Thermalizes and measures B parallel chains with automatic step-size tuning during
        the first 60% of thermalization (strictly frozen during measurement to preserve
        exact detailed balance!).
        """
        B = X_init.shape[0]
        X = self._hermitianize(X_init.clone().to(self.device))
        if U_init is None:
            U = self.eye_N.unsqueeze(0).expand(B, self.N, self.N).clone()
        else:
            U = U_init.clone().to(self.device)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        dt_cur = float(dt)
        adapt_end = int(0.55 * n_therm)

        # Thermalization (adapt during first 55%, then strictly fixed-dt burn-in for remaining 45%)
        for it in range(n_therm):
            X, U, acc, _ = self.hmc_step(X, U, dt_cur, n_leapfrog, gen)
            if adapt_dt and it < adapt_end and (it + 1) % 6 == 0:
                rate = float(torch.mean(acc.to(torch.float64)))
                if rate < 0.62:
                    dt_cur = max(dt_cur * 0.85, 5e-4)
                elif rate > 0.86:
                    dt_cur = min(dt_cur * 1.18, 0.25)

        # Measurement (strictly fixed dt_cur -> exact detailed balance)
        acc_list = []
        trX2_chains = []
        Evir_chains = []
        Vflux_chains = []
        sd_chains = []
        poly_chains = []
        alg_err_chains = []
        r2_spread_chains = []

        for it in range(n_meas * meas_every):
            X, U, acc, _ = self.hmc_step(X, U, dt_cur, n_leapfrog, gen)
            acc_list.append(float(torch.mean(acc.to(torch.float64))))
            if (it + 1) % meas_every == 0:
                obs = self.action_and_observables(X, U)
                trX2_chains.append(obs["mean_trX2"].cpu().numpy())
                Evir_chains.append(obs["E_virial"].cpu().numpy())
                Vflux_chains.append(obs["mean_V_flux"].cpu().numpy())
                sd_chains.append(obs["sd_lhs"].cpu().numpy())
                poly_chains.append(obs["polyakov_abs"].cpu().numpy())

                # Gauge-invariant su(2) fuzzy-sphere algebraic fidelity on slice t=0
                X0 = X[:, 0]  # (B, 3, N, N)
                tr0 = torch.real(torch.sum(X0 * X0.conj(), dim=(1, 2, 3)))
                num_a = torch.zeros(B, dtype=torch.float64, device=self.device)
                den_a = torch.zeros(B, dtype=torch.float64, device=self.device)
                for I, J, K, sgn in EPS3_PAIRS:
                    C_IJ = -1j * (X0[:, I] @ X0[:, J] - X0[:, J] @ X0[:, I])
                    num_a = num_a + torch.real(torch.einsum("bij,bji->b", C_IJ, sgn * X0[:, K]))
                    den_a = den_a + torch.real(torch.sum(X0[:, K] * X0[:, K].conj(), dim=(1, 2)))
                a_fit = num_a / torch.clamp(den_a, min=1e-12)
                alg_res = torch.zeros(B, dtype=torch.float64, device=self.device)
                for I, J, K, sgn in EPS3_PAIRS:
                    C_IJ = -1j * (X0[:, I] @ X0[:, J] - X0[:, J] @ X0[:, I])
                    diff = C_IJ - (a_fit * sgn).view(B, 1, 1) * X0[:, K]
                    alg_res = alg_res + torch.real(torch.sum(diff * diff.conj(), dim=(1, 2)))
                rel_alg = torch.sqrt(alg_res / torch.clamp(2.0 * (a_fit ** 2) * tr0, min=1e-12))
                alg_err_chains.append(rel_alg.cpu().numpy())

                R2_0 = X0[:, 0] @ X0[:, 0] + X0[:, 1] @ X0[:, 1] + X0[:, 2] @ X0[:, 2]
                R2_0 = self._hermitianize(R2_0)
                r2_ev = torch.linalg.eigvalsh(R2_0).real  # (B, N)
                sp = (r2_ev[:, -1] - r2_ev[:, 0]) / torch.clamp(torch.mean(r2_ev, dim=1), min=1e-12)
                r2_spread_chains.append(sp.cpu().numpy())

        def summarize(arr_list):
            # arr_list: (n_meas, B). Compute chain means first so SE is strictly independent across B chains!
            A = np.array(arr_list, dtype=float)
            chain_means = np.mean(A, axis=0)
            mean = float(np.mean(chain_means))
            se = float(np.std(chain_means, ddof=1) / np.sqrt(B)) if B > 1 else float(np.std(A) / np.sqrt(len(A)))
            return mean, se

        trX2_m, trX2_se = summarize(trX2_chains)
        Evir_m, Evir_se = summarize(Evir_chains)
        Vfl_m, Vfl_se = summarize(Vflux_chains)
        sd_m, sd_se = summarize(sd_chains)
        poly_m, poly_se = summarize(poly_chains)
        alg_m, alg_se = summarize(alg_err_chains)
        sp_m, sp_se = summarize(r2_spread_chains)
        sd_exact = float(3 * (self.N ** 2) * self.L)

        return {
            "N": self.N, "L": self.L, "beta": self.beta, "a": self.a,
            "g": self.g, "mu": self.mu, "nu": self.nu, "gauged": self.gauged,
            "n_chains": B, "dt_final": dt_cur,
            "accept_rate": float(np.mean(acc_list)),
            "trX2_mean": trX2_m, "trX2_se": trX2_se,
            "E_virial_mean": Evir_m, "E_virial_se": Evir_se,
            "V_flux_mean": Vfl_m, "V_flux_se": Vfl_se,
            "alg_err_mean": alg_m, "alg_err_se": alg_se,
            "r2_spread_mean": sp_m, "r2_spread_se": sp_se,
            "polyakov_mean": poly_m, "polyakov_se": poly_se,
            "sd_lhs_mean": sd_m, "sd_lhs_se": sd_se, "sd_exact": sd_exact,
            "sd_zscore": abs(sd_m - sd_exact) / max(sd_se, 1e-12),
            "X_final": X, "U_final": U,
        }
