import warnings
from numbers import Real
from pathlib import Path
from typing import Tuple, Iterable

import sqlite3

from .import_utils import np

# Setup default database
db_dir = Path.home() / ".photon_canon"
db_dir.mkdir(parents=True, exist_ok=True)
db_path = db_dir / "lut.db"
CON = sqlite3.connect(db_path)
c = CON.cursor()

try:
    c.execute("SELECT max(id) FROM mclut_simulations")
    latest_simulation_id = c.fetchone()[0]
except sqlite3.OperationalError:
    latest_simulation_id = None
    warnings.warn('No default LUT found. Simulate one if you have not.')


def simulate(system: "System", n: int, **kwargs) -> Tuple[float, float, float]:
    photons = system.beam(n=n, **kwargs)
    photons.simulate()
    return photons.T, photons.R, photons.A


def sample_spectrum(wavelengths: Iterable[Real],
                    spectrum: Iterable[Real]):
    wavelengths = np.asarray(wavelengths)
    spectrum = np.asarray(spectrum)

    # Normalize PDF
    spectrum /= np.sum(spectrum)

    # Compute CDF
    cdf = np.cumsum(spectrum)

    # Take random sample
    i = np.random.uniform(0, 1)

    # Interpolate value of sample from CDF
    return np.interp(i, cdf, wavelengths)
