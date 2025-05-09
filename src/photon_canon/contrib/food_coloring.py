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

# Original dilution was 45 uL of stock (5uL dye in 3mL water) into 3.4 mL of water
s = (
    45 / 1000 * 5 / 3 / 3.4  # uL -> mL  # stock concentration  # cuvette volume
)  # concentration measured in uL / mL
s = 1 / s  # invert to scale to 1uL / mL

# Normalize units to 1 uL / mL
a_red *= s
a_yellow *= s
a_green *= s
a_blue *= s

# Load pre-mixed validation stocks.
# Stock A: 0.125 mL / mL red, 0.175 mL / mL green in water
# Stock B: 0.600 mL / mL yellow, 0.050 mL / mL blue in water
# Stocks were diluted down for absorbance measurements to 10 uL/mL stock
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

    # Raw colors
    if wavelengths is None:
        wl = raw_wl
    else:
        wl = wavelengths

    mask = raw_wl.isin(wl)
    a_r = a_red[mask].values
    a_y = a_yellow[mask].values
    a_g = a_green[mask].values
    a_b = a_blue[mask].values

    # Premixed
    if wavelengths is None:
        wl = premix_wl

    mask = premix_wl.isin(wl)
    a_sa = a_stock_a[mask].values
    a_sb = a_stock_b[mask].values

    return np.array(
        red * a_r
        + yellow * a_y
        + green * a_g
        + blue * a_b
        + stock_a * a_sa
        + stock_b * a_sb
    )
