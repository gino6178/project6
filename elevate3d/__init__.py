"""Elevate3D — indoor scene synthesis with non-planar floors."""
import os as _os, sys as _sys

# ReRoom (project5) supplies the 3D-FRONT parser, category table, asset bank and
# the 2D geometry helpers.  It is imported, not vendored, so fixes flow one way.
REROOM_ROOT = _os.environ.get("REROOM_ROOT", "/home/gino/project/project5")
if REROOM_ROOT not in _sys.path and _os.path.isdir(REROOM_ROOT):
    _sys.path.insert(0, REROOM_ROOT)

__version__ = "0.1.0"
