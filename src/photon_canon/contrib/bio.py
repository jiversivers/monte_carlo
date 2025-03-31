import importlib.resources
from numbers import Real
from typing import Union, Iterable, Tuple

import numpy as np
import pandas as pd
from numpy._typing import NDArray

# Read data file for function usage
with importlib.resources.open_text('photon_canon.data', "hbo2_hb.tsv") as f:
    df = pd.read_csv(f, sep='\t', skiprows=1)
wl, hbo2, dhb = df['lambda'], df['hbo2'], df['hb']
wl = np.array([float(w) for w in wl[1:]])
hbo2 = np.array([float(h) for h in hbo2[1:]])
dhb = np.array([float(h) for h in dhb[1:]])
eps = np.stack((hbo2, dhb))

# Calculate some defaults (good for initial guesses)
tHb = 4
tHb /= 64500  # molar mass of hemoglobin
sO2 = 0.98


def calculate_mus(a: Real = 1,
                  b: Real = 1,
                  ci: Union[Real, Iterable[Real]] = (tHb * sO2, tHb * (1 - sO2)),
                  epsilons: Union[Iterable[Real], Iterable[Iterable[Real]]] = eps,
                  wavelength: Union[Real, Iterable[Real]] = wl,
                  wavelength0: Real = 650,
                  force_feasible: bool = True) -> Union[Tuple[Real, Real, Real], Tuple[NDArray, NDArray, NDArray]]:
    # Check cs and epsilons match up
    msg = ('One alpha must be included for all species, but you gave {} ci and {} spectra. '
           'In the case of only two species, the second alpha may be omitted')
    try:
        # Simple 1 to 1 ratio of multiple in list-likes
        if isinstance(ci, (list, tuple, np.ndarray)):
            assert len(ci) == len(epsilons), AssertionError(msg.format(len(ci), len(wavelength)))
        # or 1 ci and either a single list-like OR a one element list-like where that element is list-like
        elif isinstance(ci, (int, float)):
            if isinstance(epsilons[0], (list, tuple, np.ndarray)):
                assert len(epsilons) == 1, AssertionError(msg.format(1, len(epsilons)))

        # Check cs make sense
        if force_feasible:
            msg = 'Concentrations cannot be negative'
            if isinstance(ci, (list, tuple, np.ndarray)):
                assert np.all(np.array([c >= 0 for c in ci])), AssertionError(msg)
            elif isinstance(ci, (int, float)):
                assert ci >= 0, AssertionError(msg)

        # Check that wavelengths and epsilons match up
        msg = (f'A spectrum of molar absorptivity must be included with each spectrum. '
               f'You gave {len(wavelength)} wavelengths but molar absorptivity had {len(epsilons[0])} elements.')
        # Either each element of the epsilons has its own element for the wavelengths
        if isinstance(epsilons[0], (list, tuple, np.ndarray)):
            assert np.all(np.array([len(e) == len(wavelength) for e in epsilons])), AssertionError(msg)
        # Or there is only one species, and it has its own elements for all wavelengths
        elif isinstance(epsilons[0], (int, float)):
            assert len(epsilons) == len(wavelength), AssertionError(msg)

    except AssertionError as e:
        raise ValueError(e)

    wavelength = np.asarray(wavelength)  # Wavelengths of measurements (nm)
    mu_s = a * (wavelength / wavelength0) ** -b  # Reduced scattering coefficient, cm^-1

    # Unpack list of spectra (if it is a list)
    if isinstance(epsilons[0], (tuple, list, np.ndarray)):
        epsilons = np.asarray([np.asarray(spectrum) for spectrum in epsilons])  # Molar absorptivity (L/(mol cm))
    else:
        epsilons = np.asarray(epsilons)

    # Reshape concentrations (if multiple)
    if isinstance(ci, (list, tuple, np.ndarray)):
        ci = np.asarray(ci)
        ci = ci.reshape(-1, 1)

    mu_a = np.log(10) * np.sum(ci * epsilons, axis=0)  # Absorption coefficient, cm^-1
    return mu_s, mu_a, wl
