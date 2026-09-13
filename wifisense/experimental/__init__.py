"""Experimental CSI-only modules (breathing / heartbeat).

Explicitly separated from the core pipeline so the main detection chain has no
dependency on anything here. Active only when the sensing mode is ``csi``.
"""

from . import breathing

__all__ = ["breathing"]
