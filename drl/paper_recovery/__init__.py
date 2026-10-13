"""Standalone Q-Relay paper-recovery controller package.

Historical versioned environments and checkpoints are intentionally not imported
here.  The public surface is the frozen recovery configuration, environment,
residual policy, and receipt helpers.
"""

from .config import PaperRecoveryConfig
from .env import PaperRecoveryEnv

__all__ = ["PaperRecoveryConfig", "PaperRecoveryEnv"]
