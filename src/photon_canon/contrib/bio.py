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
    """Helper function to check if concentrations are valid (non-negative)

    :param ci: The concentration of absorbers 1-i. (default: (0.5, 0.5))
    :type ci: Iterable[Real]
    :param epsilons: The extinction coefficient for absorbers 1-i (default: Hb and HbO2 absorption).
    :type epsilons: Iterable[Real]
    :param wavelength: Array of wavelengths (in nm) at which to compute the values (default: [250, 1000], step 2).
    :type wavelength: Iterable[Real]
    :return: None. Raises error if concentrations are invalid.
    :rtype: None
    :raises ConcentrationError: if concentrations are invalid
    """

    def check_concentrations_valid():
        if isinstance(ci, (list, tuple, np.ndarray)):
            if np.any(np.array(ci) < 0):
                raise ConcentrationError("Concentrations cannot be negative")
        elif isinstance(ci, (int, float)):
            if ci < 0:
                raise ConcentrationError("Concentrations cannot be negative")

    # Check if lengths match for list-like concentrations and epsilons
    def check_lengths_match():
        if isinstance(ci, (list, tuple, np.ndarray)):
            if len(ci) != len(epsilons):
                raise OpticalPropertyError(
                    f"Length mismatch: {len(ci)} concentrations vs {len(epsilons)} epsilons"
                )
        elif isinstance(ci, (int, float)):
            if (
                isinstance(epsilons[0], (list, tuple, np.ndarray))
                and len(epsilons) != 1
            ):
                raise OpticalPropertyError(
                    f"Expected 1 epsilon, but got {len(epsilons)}"
                )

    # Check if wavelengths and epsilons match up
    def check_wavelength_epsilon_match():
        msg = f"A spectrum of molar absorptivity must be included with each spectrum. You gave {len(wavelength)} wavelengths but molar absorptivity had {len(epsilons[0])} elements."
        if isinstance(epsilons[0], (list, tuple, np.ndarray)):
            if not all(len(e) == len(wavelength) for e in epsilons):
                raise OpticalPropertyError(msg)
        elif isinstance(epsilons[0], (int, float)) and len(epsilons) != len(wavelength):
            raise OpticalPropertyError(msg)

        check_lengths_match(ci, epsilons)

        check_wavelength_epsilon_match(wavelength, epsilons)

        check_concentrations_valid(ci)


def inverse_power_law_for_reduced_scattering(
    a: Real = 1,
    b: Real = 1,
    wavelength: Union[Real, Iterable[Real]] = None,
    wavelength0: Real = 650,
) -> Union[Real, Iterable[Real]]:
    r"""Calculate the reduced scatter (:math:`\mu_s'`) as

    .. math::

        \mu_s' = a\left(\frac{\lambda}{\lambda_0}\right)^{-b}


    :param a: Scattering amplitude for the reduced scattering coefficient (default: 1 cm^-1).
    :type a: Real
    :param b: Scattering power exponent (default: 1).
    :type b: Real
    :param wavelength: Array of wavelengths (in nm) at which to compute the values (default: [250, 1000], step 2).
    :type wavelength: Iterable[Real]
    :param wavelength0: Reference wavelength for scattering.
    :type wavelength0: Real
    :return: Reduced scattering coefficient
    :rtype: Real
    """
    if wavelength is None:
        wavelength = wl
    mu_s = a * (wavelength / wavelength0) ** -b  # Reduced scattering coefficient, cm^-1
    return mu_s


def spectroscopic_model_of_absorption(
    ci: Union[Real, Iterable[Real]] = (tHb * sO2, tHb * (1 - sO2)),
    epsilons: Union[Iterable[Real], Iterable[Iterable[Real]]] = eps,
    wavelength: Union[Real, Iterable[Real]] = None,
    force_feasible: bool = False,
) -> Union[Real, Iterable[Real]]:
    """Calculate the absorption coefficient (:math:`\mu_a`) as

    .. math::

        \mu_a = \sum_i\epsilon_iC_i

    :param ci: The concentration of absorbers 1-i. (default: (0.5, 0.5))
    :type ci: Iterable[Real]
    :param epsilons: The extinction coefficient for absorbers 1-i (default: Hb and HbO2 absorption).
    :type epsilons: Iterable[Real]
    :param wavelength: Array of wavelengths (in nm) at which to compute the values (default: [250, 1000], step 2).
    :type wavelength: Iterable[Real]
    :param force_feasible: Whether to force feasible concentrations (non-negative) (default: True).
    :type force_feasible: bool
    :return: Absorption coefficient of mixture of absorber at input wavelengths.
    :rtype: Iterable[Real]
    """
    if wavelength is None:
        wavelength = wl

    # Check cs and epsilons match up and are feasible
    try:
        validate_concentrations_and_spectra(ci, epsilons, wavelength)
    except ConcentrationError as e:
        if force_feasible:
            # Stash infeasible results into error
            e.stashed = (
                f"Results of calculation with `force_feasible=False`: "
                f"{spectroscopic_model_of_absorption(ci, epsilons, wavelength, force_feasible=False)}"
            )
            raise e

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
    return mu_a


def calculate_mus(
    a: Real = 1,
    b: Real = 1,
    ci: Union[Real, Iterable[Real]] = (tHb * sO2, tHb * (1 - sO2)),
    epsilons: Union[Iterable[Real], Iterable[Iterable[Real]]] = eps,
    wavelength: Union[Real, Iterable[Real]] = None,
    wavelength0: Real = 650,
    force_feasible: bool = True,
) -> Union[Tuple[Real, Real, Real], Tuple[NDArray, NDArray, NDArray]]:
    """Calculate the reduced scatter (μs') and absorption (μa) coefficients. See
      :py:func:`inverse_power_law_for_reduced_scattering` and :py:func:`spectroscopic_model_of_absorption` for
      mathematical details. Coefficients are calculated from input parameters a, b, c and epsilon over the
      wavelengths provided.

    :param a: Scattering amplitude for the reduced scattering coefficient (default: 1 cm^-1).
    :type a: Real
    :param b: Scattering power exponent (default: 1).
    :type b: Real
    :param ci: The concentration of absorbers 1-i. (default: (0.5, 0.5))
    :type ci: Iterable[Real]
    :param epsilons: The extinction coefficient for absorbers 1-i (default: Hb and HbO2 absorption).
    :type epsilons: Iterable[Real]
    :param wavelength: Array of wavelengths (in nm) at which to compute the values (default: [250, 1000], step 2).
    :type wavelength: Iterable[Real]
    :param wavelength0: Reference wavelength for scattering.
    :type wavelength0: Real
    :param force_feasible: Whether to force feasible concentrations (non-negative) (default: True).
    :type force_feasible: bool
    :raises OpticalPropertyError: When concentrations and epsilons are incompatible, or if force_feasible is True
        concentrations are negative .
    :return: Reduced scattering coefficient and absorption coefficient.
    :rtype: Tuple[NDArray, NDArray, NDArray]
    """
    if wavelength is None:
        wavelength = wl

    # Array everything as needed
    wavelength = np.asarray(wavelength)  # Wavelengths of measurements (nm)

    return (
        inverse_power_law_for_reduced_scattering(a, b, wavelength, wavelength0),
        spectroscopic_model_of_absorption(
            ci, epsilons, wavelength, force_feasible=force_feasible
        ),
        wavelength,
    )


def hemoglobin_mus(
    a: Real = 1,
    b: Real = 1,
    t: Real = tHb,
    s: Real = sO2,
    wavelengths: Iterable[Real] = None,
    force_feasible: bool = True,
) -> Union[Tuple[Real, Real, Real], Tuple[NDArray, NDArray, NDArray]]:
    """
    Computes the reduced scattering coefficient (μs') for hemoglobin solutions
    based on given absorption coefficients of oxyhemoglobin (HbO2) and deoxyhemoglobin (Hb). See :py:func:`calculate_mus`
    for more details.

    Default values are supplied via Scott Prahl at https://omlc.org/spectra/hemoglobin/summary.html

    This function interpolates the extinction coefficients of HbO2 and Hb
    at specified wavelengths using cubic interpolation and calculates the
    corresponding μs' values.

    :param force_feasible: Option to prevent impossible values (like negative concentrations) (default: True)
    :type force_feasible: bool
    :param a: Scattering amplitude for the reduced scattering coefficient (default: 1).
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
    if wavelengths is None:
        wavelengths = wl
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
    """Forwarding function to streamline :math:`\mathrm{[tHb]\,\&\,\mathrm{sO_2}` modelling. This function takes biological parameters and turns them
    optical properties, then forwards those to model reflectance and returns that reflectance. In short, biological
    properties get converted to reflectance using the given model. See :py:func:`hemoglobin_mus` and
    :py:func:`calculate_mus` for more details.

    :param model: Function that takes a wavelength array and returns a reflectance.
    :type model: Callable[[float, float, ...], float]
    :param wavelengths: Array of wavelengths (in nm) at which to compute the values (in nm).
    :type wavelengths: Iterable[Real]
    :param a: Reduced scattering amplitude.
    """
    mu_s, mu_a, _ = hemoglobin_mus(a, b, t, s, wavelengths)
    return model_reflectance(model, mu_s, mu_a, **kwargs)
