"""Modular experiment suites for Project Genesis-Rigorous."""

from .phase1_emergence import run_phase1_experiment
from .phase2_gravity import run_phase2_gravity_experiment
from .phase3a_dynamics import run_phase3a_dynamics_experiment
from .phase3b_large_n import run_phase3b_large_n_experiment
from .phase3c_gauge import run_phase3c_gauge_experiment
from .phase3d_chiral import run_phase3d_chiral_experiment

__all__ = [
    "run_phase1_experiment",
    "run_phase2_gravity_experiment",
    "run_phase3a_dynamics_experiment",
    "run_phase3b_large_n_experiment",
    "run_phase3c_gauge_experiment",
    "run_phase3d_chiral_experiment",
]
