import importlib.resources

with importlib.resources.open_text('photon_canon.data', "sample_food_coloring_absorption.csv") as f:
    df = pd.read_csv(f)
wl, a_red, a_yellow, a_green, a_blue = df['Wavelength'], df['red_45ul'], df['yellow_45ul'], df['green_45ul'], df['blue_45ul']

# Normalize units to 1 uL/mL
a_red *= 3.4/45
a_yellow *= 3.4/45
a_green *= 3.4/45
a_blue *= 3.4/45

def make_mix(*, red=0, yellow=0, green=0, blue=0):
    return np.array(red * a_red + yellow * a_yellow + green * a_green + blue * a_blue)