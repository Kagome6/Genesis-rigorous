"""
Project Genesis-Rigorous: Unified Emergent Spacetime, Gravity, Gauge & Chiral Matter Suite.
"""

__version__ = "1.0.0"

from .safety import HardwareSafetyGuard, SafetyConfig, STOP_REQUESTED, safe_eigh, safe_eigvalsh
from .exact_hfb import GenesisQuantumCore, analyze_matrix_geometry, build_hermitian_basis, build_su2_irrep
from .matrix_native import MatrixNativeQuantumCore
from .dirac_matter import AdjointDiracMatterSector, find_shell_radius_along_ray, scan_meridian_geometry

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
    "analyze_matrix_geometry",
    "build_hermitian_basis",
    "build_su2_irrep",
    "find_shell_radius_along_ray",
    "scan_meridian_geometry",
]
