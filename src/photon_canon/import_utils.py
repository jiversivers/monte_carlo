try:
    import cupy as np
    from cupy import iterable
    from cupy.typing import NDArray
    from cupyx.scipy.interpolate import RegularGridInterpolator

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError):
    import numpy as np
    from numpy import iterable
    from numpy.typing import NDArray
    from scipy.interpolate import RegularGridInterpolator

    np.is_available = lambda: False  # Mock the `is_available` method for consistenc
