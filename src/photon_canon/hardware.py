import functools
from numbers import Real
from typing import Tuple, Union, Callable, Optional, Iterable
from .import_utils import np, iterable, NDArray

# Hardware specs in cm
ID = 0.16443276801785274
OD = 0.3205672319821467
WD = 0.2

THETA = np.arctan(-OD / WD)  # rad
NA = 1.0

# Create random number generator
rng = np.random.default_rng()

# Default "sampler" and "detector" to start photons straight down at origin and detect all reflected
def pencil_beams(n: int) -> Tuple[NDArray[float], NDArray[float]]:
    return (np.repeat(np.array((0, 0, 0))[np.newaxis, ...], n, axis=0),
            np.repeat(np.array((0, 0, 1))[np.newaxis, ...], n, axis=0))


def hollow_cone_beam(n: int = 50000,
                     r_min: float = ID,
                     r_max: float = OD,
                     angle_min: float = THETA,
                     angle_max: float = THETA) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Sample angle and radius for starting location
    phi = np.random.uniform(0, 2 * np.pi, n)
    r = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2, n))

    # Create ring (2D coordinates)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    location = np.vstack((x, y, np.zeros(n))).T  # Stack x, y, and z=0 into shape (n, 3)

    # Sample injection angles directional cosines
    theta = np.random.uniform(angle_min, angle_max, n)

    # Compute directional cosines
    mu_x = np.sin(theta) * np.cos(phi)
    mu_y = np.sin(theta) * np.sin(phi)
    mu_z = np.cos(theta)
    directional_cosines = np.vstack((mu_x, mu_y, mu_z)).T  # Stack into shape (n, 3)

    return location, directional_cosines


# NOTE: This samples the ring _at_ where ID/OD are measured from, presumably a distance WD above the medium of measure.
def create_hollow_cone_beam(r_bounds: Union[Real, Tuple[Real, Real]],
                            angle_bounds: Union[Real, Tuple[Real, Real]]) -> Callable:
    if not iterable(r_bounds):
        r_max = r_bounds
        r_min = 0
    elif len(r_bounds) == 1:
        r_max = r_bounds[0]
        r_min = 0
    else:
        r_min, r_max = r_bounds

    if not iterable(angle_bounds):
        angle_min = angle_bounds
        angle_max = angle_bounds
    elif len(angle_bounds) == 1:
        angle_min = angle_bounds[0]
        angle_max = angle_bounds[0]
    else:
        angle_min, angle_max = angle_bounds

    return functools.partial(hollow_cone_beam, r_min=r_min, r_max=r_max, angle_min=angle_min, angle_max=angle_max)


def oblique_beams(n: int,
                  beams: Tuple[int, ...] = (0, 1),
                  gamma: float = 75,
                  w: float = 1):
    c = np.radians(gamma)

    # Sample radial positions within a circular beam profile
    r = w * np.sqrt(np.random.uniform(0, 1, n))  # Uniform distribution in area
    phi = np.random.uniform(0, 2 * np.pi, n)     # Uniform angular distribution

    # Calculate the offsets based on the radial distance and angle
    in_plane_offsets = r * np.cos(phi)
    out_of_plane_offsets = r * np.sin(phi)

    # Calculate incidence point on horizontal plane
    in_plane_incidence = in_plane_offsets + (r * np.sin(c)) - (w * np.sin(c) / 2)
    in_plane_incidence /= np.cos(c)
    out_of_plane_incidence = out_of_plane_offsets  # Out of plane location is unchanged

    # Select beam and input final outputs accordingly (beam0: xz plane; beam1: yz plane)
    beam = np.random.choice(beams, n) if len(beams) > 1 else beams[0]

    location_coordinates = np.zeros((n, 3))
    directional_cosines = np.cos(c) * np.ones((n, 3))
    for i in beams:
        location_coordinates[:, i] = np.where(beam == i, in_plane_incidence, out_of_plane_incidence)
        directional_cosines[:, i] = np.where(beam == i, np.sin(c), 0)

    return location_coordinates, directional_cosines


def create_oblique_beams(beams: Tuple[int, ...] = (0, 1),
                         gamma: float = 75,
                         w: float = 1):
    """
    Factory functon to create sampler. Just prefills and freezes the set parameters adn returns a callable that will
    sample n samples with the pre-provided fixed parmeters.

    NOTE: Using functools.partial makes this pickleable for multiprocessing as opposed to defining the sampler function
    locally inside this factory.

    :param beams: Tuple of indices for which beams are "on".
    :param gamma: Angle in degrees from the z-axis.
    :param w: Width of the beam.
    :return: Frozen sampler function with the parameters set here.
    """
    return functools.partial(oblique_beams, beams=beams, gamma=gamma, w=w)



# Stand-in detector for all queried
def total_acceptor(x: float, y: float, mu_z: Optional[float] = None) -> NDArray[np.bool_]:
    return np.full_like(x, fill_value=True, dtype=np.bool_)

def cone_of_acceptance(x: Union[Real, Iterable[Real]],
                       y: Union[Real, Iterable[Real]],
                       mu_z: Union[Real, Iterable[Real]] = None,
                       na: float = NA,
                       n: float = 1.33,
                       r: float = ID) -> NDArray[np.bool_]:
    x = np.array(x)
    y = np.array(y)
    if mu_z is not None:
        theta_max = np.arcsin(na / n)
        mu_z_max = np.cos(theta_max)
        too_steep = np.array(mu_z) > mu_z_max
    else:
        too_steep = False
    r_test = np.sqrt(x ** 2 + y ** 2)
    outside = r_test > r
    return ~too_steep & ~outside


def create_cone_of_acceptance(r: Real,
                              na: Real = NA,
                              n: Real = 1.33) -> Callable:
    return functools.partial(cone_of_acceptance, r=r, na=na, n=n)
