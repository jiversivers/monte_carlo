import warnings
from numbers import Real
from pathlib import Path
from typing import Tuple, Iterable, Union, Callable

import sqlite3

from .import_utils import np

# Setup default database
db_dir = Path.home() / ".photon_canon"
db_dir.mkdir(parents=True, exist_ok=True)
db_path = db_dir / "lut.db"
CON = sqlite3.connect(db_path)
c = CON.cursor()

class OpticalPropertyError(Exception):
    """
    Custom exception raised when invalid or undefined optical properties are encountered.

    This can be used to signal errors related to scattering or absorption values, LUT mismatches,
    or other property-related modeling issues.
    """
    pass

# This block attempts to retrieve the latest simulation ID from the mclut_simulations table.
# If the table does not exist or the query fails, it sets latest_simulation_id to None and emits a warning.
try:
    c.execute("SELECT max(id) FROM mclut_simulations")
    latest_simulation_id = c.fetchone()[0]
except sqlite3.OperationalError:
    latest_simulation_id = None
    warnings.warn("No default LUT found. Simulate one if you have not.")


def simulate(system: "System", n: int, **kwargs) -> Tuple[float, float, float]:
    """
    Runs a Monte Carlo simulation using the provided system configuration and number of photons.

    The simulation is executed on a `System` object, which models beam propagation and optical interactions.
    Results include the transmittance (T), reflectance (R), and absorption (A) coefficients.

    :param system: The optical system object capable of simulating photon transport.
    :type system: System
    :param n: Number of photons to simulate.
    :type n: int
    :param kwargs: Additional keyword arguments passed to `system.beam()`.
    :type kwargs: Any

    :return: Tuple containing transmittance (T), reflectance (R), and absorption (A).
    :rtype: Tuple[float, float, float]
    """
    photons = system.beam(n=n, **kwargs)
    photons.simulate()
    return photons.T, photons.R, photons.A


def sample_spectrum(wavelengths: Iterable[Real], spectrum: Iterable[Real]):
    """
    Samples a wavelength from a spectral probability distribution.

    The function normalizes the input spectrum to form a probability density function (PDF),
    computes the cumulative distribution function (CDF), and uses inverse transform sampling
    to randomly select a wavelength based on the distribution.

    :param wavelengths: Array-like list of wavelengths.
    :type wavelengths: Iterable[Real]
    :param spectrum: Array-like list of spectral intensities or probabilities corresponding to the wavelengths.
    :type spectrum: Iterable[Real]

    :return: A single sampled wavelength based on the input spectrum.
    :rtype: float
    """
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


def model_reflectance(
    model: Callable[[float, float, ...], Union[float, np.ndarray]],
    mu_s: np.ndarray[float],
    mu_a: np.ndarray[float],
    vectorize: bool = False,
    **kwargs
) -> np.ndarray[float]:
    """
    Generalizes reflectance modeling by applying a provided model function to arrays of scattering and absorption coefficients.

    This function computes modeled reflectance based on the given `model`, scattering coefficients (`mu_s`),
    and absorption coefficients (`mu_a`). If `vectorize` is set to True, the function assumes that the model
    supports vectorized input and calls it once for all values. Otherwise, it loops through each pair
    of (`mu_s`, `mu_a`) values and computes the reflectance individually.

    :param model: A function that accepts `mu_s`, `mu_a`, and optional keyword arguments to return reflectance.
                  Must return either a float or a NumPy array.
    :type model: Callable[[float, float, ...], Union[float, np.ndarray]]
    :param mu_s: An array of scattering coefficients.
    :type mu_s: np.ndarray[float]
    :param mu_a: An array of absorption coefficients.
    :type mu_a: np.ndarray[float]
    :param vectorize: Whether to evaluate the model in a vectorized fashion. If False, uses a loop.
    :type vectorize: bool
    :param kwargs: Additional keyword arguments to pass to the model function.
    :type kwargs: Any

    :return: Array of modeled reflectance values corresponding to each (`mu_s`, `mu_a`) pair.
    :rtype: np.ndarray[float]
    """
    if vectorize:
        rd = model(mu_s, mu_a, **kwargs)
    else:
        rd = []
        for mus, mua in zip(mu_s, mu_a):
            rd.append(model(mus, mua, **kwargs))
        rd = np.array(rd)
    return rd.squeeze()
