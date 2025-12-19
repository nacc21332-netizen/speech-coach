import sys
from pathlib import Path

# **CRITICAL**: Import real numpy from site-packages BEFORE anything else
# This prevents the local numpy.py stub from being imported
import importlib.util
_numpy_spec = importlib.util.find_spec("numpy")
if _numpy_spec and _numpy_spec.origin and "site-packages" in _numpy_spec.origin:
    # Real numpy is installed; remove local numpy.py from module cache
    if "numpy" in sys.modules:
        del sys.modules["numpy"]
    # Force import from site-packages
    import numpy as _
    del _

# Ensure project root is on sys.path so `import app` works during tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
