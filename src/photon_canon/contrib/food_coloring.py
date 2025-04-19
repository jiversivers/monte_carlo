with importlib.resources.open_text('photon_canon.data', "sample_food_coloring_absorption.csv") as f:
    df = pd.read_csv(f)
wl, red, green, blue = df['Wavelength'], df['red_45ul'], df['green_45ul'], df['blue_45ul']