"""
Project Genesis-Rigorous v1.0.2: U(N)-Covariant Mean-Field + Exact HMC
PyTorch/CUDA Testbed for Matrix Quantum Mechanics & Subregion Entanglement.
"""

__version__ = "1.0.2"

from .safety import HardwareSafetyGuard, SafetyConfig, STOP_REQUESTED, safe_eigh, safe_eigvalsh
from .exact_hfb import GenesisQuantumCore, analyze_matrix_geometry, build_hermitian_basis, build_su2_irrep
from .matrix_native import MatrixNativeQuantumCore
from .dirac_matter import AdjointDiracMatterSector, find_shell_radius_along_ray, scan_meridian_geometry
from .hmc_singlet import ExactSingletMQMHMC

__all__ = [
    "__version__",
    "HardwareSafetyGuard",
    "SafetyConfig",
    "STOP_REQUESTED",
    "safe_eigh",
    "safe_eigvalsh",
    "GenesisQuantumCore",
    "MatrixNativeQuantumCore",
    "AdjointDiracMatterSector",
    "ExactSingletMQMHMC",
    "analyze_matrix_geometry",
    "build_hermitian_basis",
    "build_su2_irrep",
    "find_shell_radius_along_ray",
    "scan_meridian_geometry",
]
