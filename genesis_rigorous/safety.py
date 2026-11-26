"""
Stage 0: 3-Layer Hardware Safety Guard & Numerically Safe Linear Algebra Primitives.
Designed for safe execution on consumer GPUs (e.g., RTX 3060 12GB VRAM / 32GB RAM).
"""

import gc
import signal
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    import psutil
except ImportError:
    psutil = None

STOP_REQUESTED = False


def _sigint_handler(sig, frame):
    global STOP_REQUESTED
    print("\n[SAFETY] Ctrl+C (SIGINT) detected. Stopping safely after current step...")
    STOP_REQUESTED = True


def install_sigint_handler() -> None:
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except ValueError:
        # Ignore when imported outside main thread
        pass


install_sigint_handler()


def safe_eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hermitian/symmetric eigendecomposition with CPU LAPACK fallback for degenerate cuSOLVER matrices."""
    try:
        return torch.linalg.eigh(A)
    except RuntimeError:
        ew, ev = torch.linalg.eigh(A.cpu())
        return ew.to(A.device), ev.to(A.device)


def safe_eigvalsh(A: torch.Tensor) -> torch.Tensor:
    """Hermitian/symmetric eigenvalues with CPU LAPACK fallback for degenerate cuSOLVER matrices."""
    try:
        return torch.linalg.eigvalsh(A)
    except RuntimeError:
        return torch.linalg.eigvalsh(A.cpu()).to(A.device)


@dataclass
class SafetyConfig:
    max_vram_mb: float = 3072.0        # Software VRAM cap (3.0 GB out of 12 GB)
    vram_fraction: float = 0.30        # PyTorch CUDA allocator cap (~3.6 GB on 12 GB GPU)
    max_ram_mb: float = 4096.0         # Software CPU RAM watchdog (4.0 GB out of 32 GB)
    max_allowed_N: int = 16            # O(N^6) f_abc allocation ceiling (exact HFB core)
    max_allowed_N_native: int = 48     # O(N^4) Kronecker Laplacian ceiling (matrix-native core)


class HardwareSafetyGuard:
    def __init__(self, cfg: Optional[SafetyConfig] = None, prefer_cuda: bool = True):
        self.cfg = cfg or SafetyConfig()
        self.use_cuda = bool(prefer_cuda and torch.cuda.is_available())
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            try:
                torch.cuda.set_per_process_memory_fraction(self.cfg.vram_fraction, device=0)
            except Exception as e:
                print(f"[SAFETY] Skipped VRAM fraction cap: {e}")

        if psutil is None and not tracemalloc.is_tracing():
            tracemalloc.start()

    def preflight_check(self, N: int, D: int = 3, mode: str = "exact_hfb") -> Dict[str, object]:
        """Estimates peak VRAM/RAM and blocks unsafe allocations before tensor creation."""
        if N < 2:
            raise ValueError(f"[SAFETY] N={N} is too small (require N >= 2).")

        n2 = N * N
        nB = D * n2

        if mode == "exact_hfb":
            if N > self.cfg.max_allowed_N:
                raise MemoryError(
                    f"[SAFETY BLOCK] N={N} exceeds exact_hfb ceiling max_allowed_N={self.cfg.max_allowed_N} "
                    f"(O(N^6) f_abc = {(N**6 * 8) / (1024**2):.1f} MB)."
                )
            bytes_primary = (n2 ** 3) * 8 + (nB ** 2) * 8 + (n2 * N * N) * 16
        elif mode == "matrix_native":
            if N > self.cfg.max_allowed_N_native:
                raise MemoryError(
                    f"[SAFETY BLOCK] N={N} exceeds matrix_native ceiling max_allowed_N_native={self.cfg.max_allowed_N_native}."
                )
            bytes_primary = (n2 * n2) * 16 + (D * n2) * 16
        else:
            raise ValueError(f"Unknown preflight mode: {mode}")

        est_peak_mb = 6.0 * bytes_primary / (1024.0 ** 2) + 80.0
        if self.use_cuda and est_peak_mb > self.cfg.max_vram_mb:
            raise MemoryError(
                f"[SAFETY BLOCK] Estimated peak VRAM ({est_peak_mb:.1f} MB) exceeds cap ({self.cfg.max_vram_mb:.1f} MB)."
            )
        if est_peak_mb > self.cfg.max_ram_mb:
            raise MemoryError(
                f"[SAFETY BLOCK] Estimated peak RAM ({est_peak_mb:.1f} MB) exceeds cap ({self.cfg.max_ram_mb:.1f} MB)."
            )

        return {
            "N": N,
            "D": D,
            "nB": nB,
            "mode": mode,
            "est_peak_mb": est_peak_mb,
            "device": str(self.device),
        }

    def reset_peak_vram(self) -> None:
        """Resets CUDA peak memory statistics before profiling a benchmark step."""
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats(0)

    def get_peak_vram_mb(self) -> float:
        """Returns true peak CUDA allocated VRAM in MiB since last reset."""
        if self.use_cuda:
            torch.cuda.synchronize()
            return float(torch.cuda.max_memory_allocated(0) / (1024.0 ** 2))
        return 0.0

    def check_runtime(self) -> Tuple[float, float]:
        """Checks live RSS RAM and live post-step CUDA allocated VRAM (MiB); use get_peak_vram_mb() for true peak."""
        if psutil is not None:
            ram_mb = psutil.Process().memory_info().rss / (1024.0 ** 2)
        else:
            curr, _ = tracemalloc.get_traced_memory()
            ram_mb = curr / (1024.0 ** 2)

        vram_mb = 0.0
        if self.use_cuda:
            vram_mb = torch.cuda.memory_allocated(0) / (1024.0 ** 2)
            if vram_mb > self.cfg.max_vram_mb:
                torch.cuda.empty_cache()
                raise MemoryError(
                    f"[SAFETY STOP] Runtime VRAM ({vram_mb:.1f} MB) exceeded cap ({self.cfg.max_vram_mb:.1f} MB)."
                )

        if ram_mb > self.cfg.max_ram_mb:
            gc.collect()
            raise MemoryError(
                f"[SAFETY STOP] Runtime RAM ({ram_mb:.1f} MB) exceeded cap ({self.cfg.max_ram_mb:.1f} MB)."
            )
        return ram_mb, vram_mb

    def cleanup(self) -> None:
        gc.collect()
        if self.use_cuda:
            torch.cuda.empty_cache()
