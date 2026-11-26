"""Modular experiment suites for Project Genesis-Rigorous v1.0.2."""

from .phase1_emergence import run_phase1_experiment
from .phase2_gravity import run_phase2_gravity_experiment
from .phase3a_dynamics import run_phase3a_dynamics_experiment
from .phase3b_large_n import run_phase3b_large_n_experiment
from .phase3c_gauge import run_phase3c_gauge_experiment
from .phase3d_chiral import run_phase3d_chiral_experiment
from .v102_entanglement_scaling import run_v102_entanglement_scaling_experiment
from .v102_phase_transition_hmc import run_v102_phase_transition_hmc_experiment
from .v102_membrane_control import run_v102_membrane_control_experiment
from .v102_live_observatory import run_v102_live_observatory
from .v102_spacetime_viewer import run_v102_classic_viewer

__all__ = [
    "run_phase1_experiment",
    "run_phase2_gravity_experiment",
    "run_phase3a_dynamics_experiment",
    "run_phase3b_large_n_experiment",
    "run_phase3c_gauge_experiment",
    "run_phase3d_chiral_experiment",
    "run_v102_entanglement_scaling_experiment",
    "run_v102_phase_transition_hmc_experiment",
    "run_v102_membrane_control_experiment",
    "run_v102_live_observatory",
    "run_v102_classic_viewer",
]
