import importlib.resources
from numbers import Real
from typing import Union, Iterable, Tuple, Callable

import pandas as pd

from ..import_utils import np, NDArray, interp1d
from ..utils import model_reflectance, OpticalPropertyError

# Read data file for function usage
with importlib.resources.open_text("photon_canon.data", "hbo2_hb.tsv") as f:
    df = pd.read_csv(f, sep="\t", skiprows=1)[1:]
wl, hbo2, dhb = df["lambda"], df["hbo2"], df["hb"]

# Extract arrays and convert to mu_a units (cm-1)
wl = np.array([float(w) for w in wl])
hbo2 = np.array([float(h) for h in hbo2])
dhb = np.array([float(h) for h in dhb])
eps = np.log(10) * np.stack((hbo2, dhb)) / 64500

# Calculate some defaults (good for initial guesses)
tHb = 1
sO2 = 0.5

class ConcentrationError(OpticalPropertyError):
    """Special error for concentration validation b/c it comes up all the time in Jacobian calculations for fitting"""
    pass

def validate_concentrations_and_spectra(ci, epsilons, wavelength):
    # Helper function to check if concentrations are valid (non-negative)
    def check_concentrations_valid(ci):
        if isinstance(ci, (list, tuple, np.ndarray)):
            if np.any(np.array(ci) < 0):
                raise ConcentrationError("Concentrations cannot be negative")
        elif isinstance(ci, (int, float)):
            if ci < 0:
                raise ConcentrationError("Concentrations cannot be negative")

    # Check if lengths match for list-like concentrations and epsilons
    def check_lengths_match(ci, epsilons):
        if isinstance(ci, (list, tuple, np.ndarray)):
            if len(ci) != len(epsilons):
                raise OpticalPropertyError(f"Length mismatch: {len(ci)} concentrations vs {len(epsilons)} epsilons")
        elif isinstance(ci, (int, float)):
            if isinstance(epsilons[0], (list, tuple, np.ndarray)) and len(epsilons) != 1:
                raise OpticalPropertyError(f"Expected 1 epsilon, but got {len(epsilons)}")

    # Check if wavelengths and epsilons match up
    def check_wavelength_epsilon_match(wavelength, epsilons):
        msg = f"A spectrum of molar absorptivity must be included with each spectrum. You gave {len(wavelength)} wavelengths but molar absorptivity had {len(epsilons[0])} elements."
        if isinstance(epsilons[0], (list, tuple, np.ndarray)):
            if not all(len(e) == len(wavelength) for e in epsilons):
                raise OpticalPropertyError(msg)
        elif isinstance(epsilons[0], (int, float)) and len(epsilons) != len(wavelength):
            raise OpticalPropertyError(msg)

        check_lengths_match(ci, epsilons)

        check_wavelength_epsilon_match(wavelength, epsilons)

        check_concentrations_valid(ci)

def calculate_mus(
    a: Real = 1,
    b: Real = 1,
    ci: Union[Real, Iterable[Real]] = (tHb * sO2, tHb * (1 - sO2)),
    epsilons: Union[Iterable[Real], Iterable[Iterable[Real]]] = eps,
    wavelength: Union[Real, Iterable[Real]] = wl,
    wavelength0: Real = 650,
    force_feasible: bool = True,
) -> Union[Tuple[Real, Real, Real], Tuple[NDArray, NDArray, NDArray]]:
    # Check cs and epsilons match up and are feasible
    try:
        validate_concentrations_and_spectra(ci, epsilons, wavelength)
    except ConcentrationError as e:
        if force_feasible:
            # Stash infeasible results into error
            raise e(calculate_mus(a, b, ci, epsilons, wavelength, wavelength0, force_feasible=False))

    # Array everything as needed
    wavelength = np.asarray(wavelength)  # Wavelengths of measurements (nm)
    mu_s = a * (wavelength / wavelength0) ** -b  # Reduced scattering coefficient, cm^-1

    # Unpack list of spectra (if it is a list)
    if isinstance(epsilons[0], (tuple, list, np.ndarray)):
        epsilons = np.asarray(
            [np.asarray(spectrum) for spectrum in epsilons]
        )  # Molar absorptivity (L/(mol cm))
    else:
        epsilons = np.asarray(epsilons)

    # Reshape concentrations (if multiple)
    if isinstance(ci, (list, tuple, np.ndarray)):
        ci = np.asarray(ci)
        ci = ci.reshape(-1, 1)
    mu_a = np.sum(ci * epsilons, axis=0)  # Absorption coefficient, cm^-1
    return mu_s, mu_a, wl


def hemoglobin_mus(
    a: Real = 1,
    b: Real = 1,
    t: Real = tHb,
    s: Real = sO2,
    wavelengths: Iterable[Real] = wl,
    force_feasible: bool = True,
) -> Union[Tuple[Real, Real, Real], Tuple[NDArray, NDArray, NDArray]]:
    """
    Computes the reduced scattering coefficient (μs') for hemoglobin solutions
    based on given absorption coefficients of oxyhemoglobin (HbO2) and deoxyhemoglobin (Hb).

    This function interpolates the extinction coefficients of HbO2 and Hb
    at specified wavelengths using cubic interpolation and calculates the
    corresponding μs' values.

    :param force_feasible: Option to prevent impossible values (like negative concentrations) (default: True)
    :type force_feasible: bool
    :param a: Scaling factor for the reduced scattering coefficient (default: 1).
    :type a: Real
    :param b: Scattering power exponent (default: 1).
    :type b: Real
    :param t: Total hemoglobin concentration (tHb) in g/L (default: `tHb`).
    :type t: Real
    :param s: Oxygen saturation (sO2) as a fraction (0 to 1) (default: `sO2`).
    :type s: Real
    :param wavelengths: Array of wavelengths (in nm) at which to compute the values (default: `wl`).
    :type wavelengths: Iterable[Real]

    :return: A tuple containing:
             - The reduced scattering coefficient (μs') at each wavelength.
             - The interpolated extinction coefficient for THb.
             - The wavelengths of the measures (passed unchanged from input, useful for defaulting and reusing)
    :rtype: Tuple[Real, Real, Real] or Tuple[NDArray, NDArray, NDArray]

    :notes:
        - The extinction coefficients for HbO2 and Hb are interpolated using cubic
          interpolation from a predefined dataset.
        - Extrapolation is used for wavelengths outside the dataset range.
        - The calculation assumes a power-law dependence on wavelength for scattering.
    """
    hbo2_interp = interp1d(wl, eps[0], kind="cubic", fill_value="extrapolate")
    dhb_interp = interp1d(wl, eps[1], kind="cubic", fill_value="extrapolate")
    epsilons = (hbo2_interp(wavelengths), dhb_interp(wavelengths))
    ci = (s * t, (1 - s) * t)
    return calculate_mus(
        a, b, ci, epsilons, wavelengths, wavelength0=650, force_feasible=force_feasible
    )


def model_from_hemoglobin(
    model: Callable[[float, float, ...], float],
    wavelengths: np.ndarray[Union[int, float]],
    a: np.ndarray[float],
    b: np.ndarray[float],
    t: np.ndarray[float],
    s: np.ndarray[float],
    **kwargs
) -> np.ndarray[float]:
    """Forwarding function to streamline Hb/sO2 modelling. This function takes biological parameters and turns them
    optical properties, then forwards those to model reflectance and returns that reflectance. In short, biological
    properties get converted to reflectance using the given model."""
    mu_s, mu_a, _ = hemoglobin_mus(a, b, t, s, wavelengths)
    return model_reflectance(model, mu_s, mu_a, **kwargs)
