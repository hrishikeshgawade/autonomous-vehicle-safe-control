"""
Root-level compatibility shim for Adaptive Cruise Control (ACC) utilities.

Exposes CarModel and sim_vehicle from simulation.acc_utils for autograders,
root execution, and IDE static analysis.
"""

from simulation.acc_utils import CarModel, sim_vehicle

__all__ = ["CarModel", "sim_vehicle"]
