import importlib.resources
import pandas as pd
import numpy as np

with importlib.resources.open_text('photon_canon.data', "sample_food_coloring_absorption.csv") as f:
    df = pd.read_csv(f)
raw_wl, a_red, a_yellow, a_green, a_blue = df['Wavelength'], df['red_45ul'], df['yellow_45ul'], df['green_45ul'], df['blue_45ul']

# Original dilution was 45 uL of stock (5uL dye in 3mL water) into 3.4 mL of water
s = (
    45 / 1000  # uL -> mL
    * 5 / 3  # stock concentration
    / 3.4  # cuvette volume
)  # concentration measured in uL / mL
s = 1/s  # invert to scale to 1uL / mL

# Normalize units to 1 uL / mL
a_red *= s
a_yellow *= s
a_green *= s
a_blue *= s

def make_mix(wavelengths: np.ndarray = None, *, red: float = 0, yellow: float = 0, green: float = 0, blue: float = 0) -> np.ndarray[float]:
    if wavelengths is None:
        wavelengths = raw_wl
    mask = raw_wl.isin(wavelengths)
    return np.array(red * a_red[mask] + yellow * a_yellow[mask] + green * a_green[mask] + blue * a_blue[mask])

# Load pre-mixed validation stocks.
# Stock A: 0.125 mL / mL red, 0.175 mL / mL green in water
# Stock B: 0.600 mL / mL yellow, 0.050 mL / mL blue in water
# Stocks were diluted down for absorbance measurements to 10 uL/mL stock
with importlib.resources.open_text('photon_canon.data', "validation_stock_absorbance.csv") as f:
    df = pd.read_csv(f)
premix_wl, stock_a, stock_b = df['Wavelength'], df['stock_a'], df['stock_b']

# Normalize to 1 uL / mL
stock_a /= 10
stock_b /= 10
