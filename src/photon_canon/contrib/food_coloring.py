import importlib.resources
import pandas as pd
import numpy as np

with importlib.resources.open_text(
    "photon_canon.data", "sample_food_coloring_absorption.csv"
) as f:
    df = pd.read_csv(f)
raw_wl, a_red, a_yellow, a_green, a_blue = (
    df["Wavelength"],
    df["red_45ul"],
    df["yellow_45ul"],
    df["green_45ul"],
    df["blue_45ul"],
)

# --------------------------------------------------------------------------
# Convert “45 µL stock in 3.4 mL” laboratory measurements to
# a *per-unit* concentration (1 µL dye / 1 mL solution) so the data can be
# linearly combined.
# --------------------------------------------------------------------------
s = (
    45 / 1000 * 5 / 3 / 3.4  # uL -> mL  # stock concentration  # cuvette volume
)  # concentration measured in uL / mL
s = 1 / s  # invert to scale to 1uL / mL

# Normalize units to 1 uL / mL
a_red *= s
a_yellow *= s
a_green *= s
a_blue *= s

# --------------------------------------------------------------------------
# Validation premixes
#   • Stock A = 0.125 mL/mL red  + 0.175 mL/mL green
#   • Stock B = 0.600 mL/mL yellow + 0.050 mL/mL blue
# Samples were measured at 10 µL/mL → rescale to 1 µL/mL
# --------------------------------------------------------------------------
with importlib.resources.open_text(
    "photon_canon.data", "validation_stock_absorbance.csv"
) as f:
    df = pd.read_csv(f)
premix_wl, a_stock_a, a_stock_b = df["Wavelength"], df["stock_a"], df["stock_b"]

# Normalize to 1 uL / mL
a_stock_a /= 10
a_stock_b /= 10


def make_mix(
    wavelengths: np.ndarray = None,
    *,
    red: float = 0,
    yellow: float = 0,
    green: float = 0,
    blue: float = 0,
    stock_a: float = 0,
    stock_b: float = 0
) -> np.ndarray[float]:
    """
    Generate a synthetic **absorbance spectrum** for an arbitrary mixture
    of food-coloring dyes (or their premixed stocks) by linear super-
    position of reference spectra.

    Parameters
    ----------
    wavelengths : ndarray, optional
        1-D array of wavelengths :math:`\lambda\;[\text{nm}]` at which
        the spectrum should be returned.  If *None* (default), the
        function returns values on the native sampling grid contained in
        the reference files.
    red, yellow, green, blue : float, keyword-only
        Concentrations (in **µL dye / mL solution**) of the individual
        stock dyes to mix.
    stock_a, stock_b : float, keyword-only
        Concentrations (µL/mL) of premixed *Stock A* and *Stock B*,
        where

        * :math:`\text{Stock A}=0.125\,\text{mL/mL red}
          + 0.175\,\text{mL/mL green}`
        * :math:`\text{Stock B}=0.600\,\text{mL/mL yellow}
          + 0.050\,\text{mL/mL blue}`

    Returns
    -------
    ndarray
        Absorbance spectrum :math:`A(\lambda)` with the same length and
        ordering as *wavelengths* (or the native grid).

    Notes
    -----
    The mixture obeys the Beer–Lambert law, so absorbances **add
    linearly**:

    .. math::

        A(\lambda) \;=\;
        \sum_{i} c_i\,a_i(\lambda),

    where

    * :math:`a_i(\lambda)` is the *per-unit* absorbance spectrum of dye
      *i* (1 µL / mL reference), and
    * :math:`c_i` is the requested concentration (µL/mL).

    The premixed stocks are treated exactly the same way, using their
    own per-unit spectra.

    Examples
    --------
    >>> # 0.2 µL/mL red + 0.1 µL/mL blue
    >>> A = make_mix(red=0.2, blue=0.1)
    >>> A.shape
    (401,)
    >>> # Integrated optical density between 400–700 nm
    >>> wl = np.linspace(400, 700, 301)
    >>> A_int = np.trapz(make_mix(wl, stock_a=0.05), wl)
    >>> A_int
    12.3  # doctest: +SKIP
    """
    # --------------------------------------------------------------
    # Select wavelengths for individual dye spectra
    # --------------------------------------------------------------
    if wavelengths is None:
        wl = raw_wl
    else:
        wl = wavelengths

    mask = raw_wl.isin(wl)
    a_r = a_red[mask].values
    a_y = a_yellow[mask].values
    a_g = a_green[mask].values
    a_b = a_blue[mask].values

    # --------------------------------------------------------------
    # Select wavelengths for premixed-stock spectra
    # --------------------------------------------------------------
    if wavelengths is None:
        wl = premix_wl

    mask = premix_wl.isin(wl)
    a_sa = a_stock_a[mask].values
    a_sb = a_stock_b[mask].values

    # --------------------------------------------------------------
    # Linear combination
    # --------------------------------------------------------------
    return np.array(
        red * a_r
        + yellow * a_y
        + green * a_g
        + blue * a_b
        + stock_a * a_sa
        + stock_b * a_sb
    )
