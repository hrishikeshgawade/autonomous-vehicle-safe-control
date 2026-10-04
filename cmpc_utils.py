"""
Root-level compatibility shim for Trajectory Tracking / CMPC utilities.

Exposes BicycleSim from simulation.cmpc_utils for autograders, root execution,
and IDE static analysis.
"""

from simulation.cmpc_utils import BicycleSim

__all__ = ["BicycleSim"]
