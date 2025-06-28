import math
import random

import numpy as np
import pytest

from photon_canon import System, Medium, Illumination, Detector
from photon_canon.optics import Photon, IndexableProperty
from photon_canon.hardware import (
    create_oblique_beams,
    create_cone_of_acceptance,
    ID,
    OD,
    THETA,
)


# ---------------------------------------------------------------------------
# -------------------------------  FIXTURES  --------------------------------
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def water() -> Medium:
    return Medium(n=1.33, desc="water")


@pytest.fixture(scope="session")
def tissue() -> Medium:
    return Medium(n=1.53, mu_a=5, mu_s=100, g=0.85, desc="tissue")


@pytest.fixture(scope="session")
def led_system(water: Medium, tissue: Medium) -> System:
    sampler = create_oblique_beams((ID, OD), THETA)
    led = Illumination(pattern=sampler)
    detector = Detector(create_cone_of_acceptance(ID))

    return System(
        water,
        0.2,
        tissue,
        float("inf"),
        surrounding_n=1.33,
        illuminator=led,
        detector=(detector, 0),
    )


@pytest.fixture
def photon(led_system: System) -> Photon:
    """Fresh photon for each test"""
    return Photon(
        wavelength=500,
        batch_size=100,
        system=led_system,
        directional_cosines=(0, 0, 1),
        location_coordinates=(0, 0, 0),
        weights=1.0,
        russian_roulette_constant=20,
        recurse=True,
        recursion_depth=0,
        recursion_limit=10,
        throw_recursion_error=True,
        keep_secondary_photons=True,
    )


# ---------------------------------------------------------------------------
# ------------------------------  TESTS: Photon  ----------------------------
# ---------------------------------------------------------------------------

def test_initialization(photon: Photon, led_system: System):
    # --- explicit settings --------------------------------------------------
    assert photon.wavelength == 500
    assert photon.batch_size == 100
    assert photon.system is led_system

    assert photon.directional_cosines.shape == (100, 3)
    assert np.all(photon.directional_cosines == np.array([0, 0, 1]))
    assert isinstance(photon.directional_cosines, IndexableProperty)

    assert photon.location_coordinates.shape == (100, 3)
    assert np.all(photon.location_coordinates == np.array([0, 0, 0]))

    assert photon.weights.shape == (100,)
    assert np.all(photon.weights == 1.0)

    assert photon.russian_roulette_constant == 20
    assert photon.recurse
    assert photon.recursion_depth == 0
    assert photon.recursion_limit == 10

    # --- hidden state -------------------------------------------------------
    assert photon.T == 0 and photon.R == 0 and photon.A == 0
    assert np.all(photon.tir_count == 0)
    assert np.all(np.isnan(photon.exit_location))
    assert np.all(np.isnan(photon.exit_direction))
    assert np.all(np.isnan(photon.exit_weights))
    assert photon.location_history.shape == (100, 3, 1)
    np.testing.assert_array_equal(
        photon.location_history.squeeze(), photon.location_coordinates
    )
    np.testing.assert_array_equal(
        photon.weights_history.squeeze(), photon.weights
    )
    assert not np.any(photon.cache_register)
    assert photon._medium.shape == (100,)
    assert photon.at_interface.shape == (100,)
    assert photon.secondary_photons == [] and isinstance(photon.secondary_photons, list)


def test_directional_cosines(photon: Photon):
    # fill
    photon.directional_cosines = (1, 0, 0)
    assert np.all(photon.directional_cosines == np.array([1, 0, 0]))

    # normalization
    photon.directional_cosines = (1, 1, 1)
    assert isinstance(photon.directional_cosines, IndexableProperty)
    norms = np.linalg.norm(photon.directional_cosines, axis=-1)
    np.testing.assert_allclose(norms, 1.0)
    np.testing.assert_array_equal(
        photon.directional_cosines, np.full((100, 3), 1 / math.sqrt(3))
    )

    # __setitem__ keeps unit length
    photon.directional_cosines[:, 2] = 1
    norms = np.linalg.norm(photon.directional_cosines, axis=-1)
    np.testing.assert_allclose(norms, 1.0)


def test_location_coordinates(photon: Photon):
    # fill
    photon.location_coordinates = (0, 0, 0.1)
    np.all(np.all(
        photon.location_coordinates == np.array([0, 0, 0.1]), axis=1
    ))

    # batch-set
    locs = np.zeros((100, 3))
    photon.location_coordinates = locs
    np.testing.assert_array_equal(photon.location_coordinates, locs)

    # indexing
    photon.location_coordinates[:, 2] = 1
    assert np.all(np.all(
        photon.location_coordinates == np.array([0, 0, 1]), axis=1
    ))


def test_weight_and_russian_roulette(photon: Photon):
    # reset whole batch
    photon.weights = 0.5
    assert np.all(photon.weights == 0.5)

    # batch vector
    photon.weights = np.repeat(1.0, photon.batch_size)
    assert np.all(photon.weights == 1.0)

    # negative -> clipped to zero + killed
    photon.weights = -1
    assert np.all(photon.weights == 0)
    assert photon.is_terminated

    # non-zero revives
    photon.weights = 1
    assert not photon.is_terminated

    # roulette behaviour (probabilistic)
    photon.weights = 0.0001
    roulette_outcomes = {0, 0.0001 * photon.russian_roulette_constant}
    assert set(photon.weights).issubset(roulette_outcomes)

    # statistical sanity check
    eps = 0.01
    trials = 2 * int((1 / eps**2) / photon.batch_size)
    hits = 0
    for _ in range(trials):
        photon.weights = 0.0001
        hits += np.sum(photon.weights != 0) / photon.batch_size
    assert math.isclose(hits / trials, 1 / photon.russian_roulette_constant, rel_tol=0, abs_tol=eps)


def test_absorb(photon: Photon, tissue: Medium):
    photon.absorb()           # still in water
    assert np.all(photon.weights == 1)

    photon.location_coordinates = (0, 0, 0.3)  # now in tissue
    photon.absorb()
    absorbed = tissue.albedo_at(photon.wavelength)
    np.testing.assert_allclose(photon.weights, 1 - absorbed)
    assert math.isclose(photon.A, photon.batch_size * absorbed, rel_tol=1e-12)


def test_move_and_interfaces(photon: Photon, water: Medium, tissue: Medium):
    # move to first interface
    photon.move()
    np.all(np.all(
        photon.location_coordinates == np.array([0, 0, 0.2]), axis=1
    ))
    assert np.all(photon.headed_into() == tissue)
    assert np.all(photon.medium == tissue)
    assert np.all(photon.at_interface)

    # specular reflection check
    spec_ref = abs((water.n - tissue.n) / (water.n + tissue.n)) ** 2
    np.testing.assert_allclose(photon.weights, 1 - spec_ref)

    # move deeper
    photon.move(0.1)
    np.testing.assert_allclose(photon.location_coordinates[:, 2], 0.3)
    assert not np.any(photon.at_interface)

    # angled incidence -> refraction
    dir_cos = np.array([1, 2, -2]) / 3
    photon.directional_cosines = dir_cos
    photon.move(float("inf"))
    n_ratio = tissue.n / water.n
    expected = dir_cos * n_ratio
    expected[2] = -math.cos(math.asin(n_ratio * math.sin(math.acos(-2 / 3))))
    np.testing.assert_allclose(photon.directional_cosines, np.repeat(expected[np.newaxis,...], 100, axis=0), rtol=1e-12, atol=1e-12)


def test_scatter(photon: Photon, tissue: Medium):
    reference = np.array([0, 0, 1])
    # start in non-scattering medium (water)
    photon.scatter()
    assert np.all(np.all(
        photon.directional_cosines == reference, axis=0
    ))

    # travel into water (still non-scatter)
    photon.move(0.1)
    photon.scatter()
    assert np.all(np.all(
        photon.directional_cosines == reference, axis=0
    ))

    # reach tissue (scatter)
    photon.move()
    photon.move(0.1)
    photon.scatter()
    assert not np.all(np.all(
        photon.directional_cosines == reference, axis=0
    ))


# ---------------------------------------------------------------------------
# ------------------------------  TESTS: Medium  ----------------------------
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def random_medium():
    props = {
        "desc": "test",
        "display_color": "gray",
        "n": random.random() + 1,
        "mu_s": 100 * random.random() + 40,
        "mu_a": 100 * random.random(),
        "g": random.random(),
    }
    return Medium(**props), props


def test_medium_init(random_medium):
    medium, props = random_medium
    for key, val in props.items():
        assert getattr(medium, key) == pytest.approx(val)


def test_mu_t(random_medium):
    medium, props = random_medium
    assert medium.mu_t == pytest.approx(props["mu_s"] + props["mu_a"])


def test_albedo(random_medium):
    medium, props = random_medium
    expected = props["mu_a"] / (props["mu_s"] + props["mu_a"])
    assert medium.albedo == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ------------------------------  TESTS: System  ----------------------------
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def basic_system() -> System:
    air = Medium(n=1.0, desc="air")
    tissue = Medium(n=1.4, mu_a=5, desc="tissue")
    water = Medium(n=1.33, desc="water")
    return System(air, 10, tissue, 20, water, 5, surrounding_n=1.0)


def test_system_initialization(basic_system: System):
    interfaces = np.asarray([0, 10, 30, 35])
    assert len(basic_system.layer) == 5
    assert basic_system.surroundings.n == 1.0
    np.testing.assert_array_equal(basic_system.boundaries[1:-1], interfaces)


@pytest.mark.parametrize(
    "z, expected",
    [
        (-5, "surroundings"),
        (0, ("surroundings", "layer[1]")),
        (5, "layer[1]"),
        (10, ("layer[1]", "layer[2]")),
        (15, "layer[2]"),
        (35, ("layer[3]", "surroundings")),
        (40, "surroundings"),
    ],
)
def test_in_medium(basic_system: System, z, expected):
    result = basic_system.in_medium(z)
    if isinstance(expected, str):
        # string is a sentinel telling us where to evaluate
        if expected == "surroundings":
            assert result is basic_system.surroundings
        elif expected.startswith("layer"):
            idx = int(expected.split("[")[1].split("]")[0])
            assert result is basic_system.layer[idx]
    else:
        left, right = result
        l_idx = int(expected[0].split("[")[1].split("]")[0]) if "layer" in expected[0] else None
        r_idx = int(expected[1].split("[")[1].split("]")[0]) if "layer" in expected[1] else None
        assert isinstance(result, tuple)
        if "surroundings" in expected[0]:
            assert left is basic_system.surroundings
        else:
            assert left is basic_system.layer[l_idx]
        if "surroundings" in expected[1]:
            assert right is basic_system.surroundings
        else:
            assert right is basic_system.layer[r_idx]


# ---------------------------------------------------------------------------
# -----------------------  Additional photon–system tests  ------------------
# ---------------------------------------------------------------------------

def test_photon_behaviour_in_simple_system():
    tissue = Medium(n=1.4, mu_s=2, mu_a=0.5, g=0.8, desc="tissue")
    water = Medium(n=1.33, mu_s=1.5, mu_a=0.3, g=0.7, desc="water")
    sys = System(tissue, 20, water, 30, surrounding_n=1.0)

    p = Photon(wavelength=500, system=sys, location_coordinates=(0, 0, 10))

    # init
    assert np.all(np.all(
        p.location_coordinates == (0, 0, 10), axis=1
    ))
    assert np.all(np.all(
        p.directional_cosines == (0, 0, 1), axis=1
    ))

    # medium transitions
    assert np.all(p.medium == tissue)
    p.location_coordinates = np.array([0, 0, 25])
    assert np.all(p.medium == water)
    p.location_coordinates = np.array([0, 0, 5])
    assert np.all(p.medium == tissue)

    # simple dynamics
    before = p.location_coordinates.copy()
    p.move()
    assert not np.array_equal(before, p.location_coordinates)

    # absorption / scattering
    before_w = p.weights
    p.absorb()
    assert np.all(np.logical_and(p.weights < before_w, p.A > 0))
    before_dir = p.directional_cosines.copy()
    p.scatter()
    assert not np.array_equal(before_dir, p.directional_cosines)

    # Russian roulette trigger
    p.weights = 0.004
    assert np.all(np.logical_or(p.weights == 0, p.weights == 0.004 * p.russian_roulette_constant))

    # Photons die
    p.weights = 0
    assert p.is_terminated

    # dead photons don't move
    before = p.location_coordinates.copy()
    p.move()
    assert np.array_equal(before, p.location_coordinates)
