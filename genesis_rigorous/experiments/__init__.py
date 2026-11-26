"""Modular experiment suites for Project Genesis-Rigorous v1.1.0 (Phase 1-3D + Live Viewer + Level-2 Law Discovery & Bootstrap)."""

from .phase1_emergence import run_phase1_experiment
from .phase2_gravity import run_phase2_gravity_experiment
from .phase3a_dynamics import run_phase3a_dynamics_experiment
from .phase3b_large_n import run_phase3b_large_n_experiment
from .phase3c_gauge import run_phase3c_gauge_experiment
from .phase3d_chiral import run_phase3d_chiral_experiment
from .law_ecosystem import run_law_ecosystem_experiment
from .dirty_law_healing import run_dirty_law_healing_experiment
from .matrix_bootstrap import run_matrix_bootstrap_experiment

__all__ = [
    "run_phase1_experiment",
    "run_phase2_gravity_experiment",
    "run_phase3a_dynamics_experiment",
    "run_phase3b_large_n_experiment",
    "run_phase3c_gauge_experiment",
    "run_phase3d_chiral_experiment",
    "run_law_ecosystem_experiment",
    "run_dirty_law_healing_experiment",
    "run_matrix_bootstrap_experiment",
]
