from photon_canon import Medium, System, Detector, Illumination
from photon_canon.hardware import create_oblique_beams
from photon_canon.import_utils import np
from photon_canon.lut import generate_lut



def main():
    # Init parameter sets
    mu_s_array = np.arange(0, 101, 1)
    mu_a_array = np.arange(1, 102, 1)
    g_array = [0.9]
    d = 0.2
    n = 100000
    tissue_n = 1.33
    surroundings_n = 1
    recurse = False
    wl0 = 650

    # Make water medium
    di_water = Medium(n=1.33, mu_s=0, mu_a=0, g=0, desc='di water', ref_wavelength=wl0)
    glass = Medium(n=1.523, mu_s=0, mu_a=0, g=0, desc='glass', ref_wavelength=wl0)

    # Create an illuminator
    lamp = Illumination(create_oblique_beams((0, 1), 60, 1.5))

    # Start the system
    system = System(di_water, 0.2,  # 1mm
                    glass, 0.017,  # 0.17mm
                    surrounding_n=surroundings_n,
                    illuminator=lamp
                    )
    tissue = Medium(n=tissue_n, mu_s=0, mu_a=0, g=1, desc='tissue')  # Placeholder to update at iteration
    system.add(tissue, d)

    # Generate a photon object (either directly or through the system illumination)
    photon = system.beam(batch_size=n, recurse=recurse)
    generate_lut(system,
                 tissue,
                 {'mu_s': mu_s_array, 'mu_a': mu_a_array, 'g': g_array},
                 photon,
                 pbar=True,
                 num_workers=1)


if __name__ == '__main__':
    main()
