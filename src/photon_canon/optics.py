from __future__ import annotations

import copy
import warnings
from numbers import Real
from typing import Optional, Iterable, Tuple, Callable, List, Any

from matplotlib import pyplot as plt
from pydantic import BaseModel, field_validator, model_validator

from .hardware import (
    create_hollow_cone_beam,
    create_cone_of_acceptance,
    ID,
    OD,
    THETA,
    pencil_beams,
    total_acceptor,
)
from .utils import sample_spectrum
from .contrib.bio import calculate_mus
from .import_utils import np, iterable, NDArray

# Get some default properties
mu_s, mu_a, wl = calculate_mus()


class Medium(BaseModel):
    r"""
    Container for the *optical properties* of a homogeneous layer (:math:`n, \mu_s, \mu_a, g`) optionally specified as
    functions of wavelength.

    :param n: Refractive index of the medium.
    :type  n: float
    :param mu_s: Scattering coefficient
                 :math:`\mu_s(\lambda)\,[\mathrm{mm}^{-1}]`
                 (scalar **or** ``numpy.ndarray``).
    :type  mu_s: float | numpy.ndarray
    :param mu_a: Absorption coefficient
                 :math:`\mu_a(\lambda)\,[\mathrm{mm}^{-1}]`
                 (scalar **or** ``numpy.ndarray``).
    :type  mu_a: float | numpy.ndarray
    :param wavelengths: Wavelength grid (nm) that matches any array
                        inputs.  Defaults to the module-level
                        ``wl`` variable.
    :type  wavelengths: numpy.ndarray
    :param ref_wavelength: Reference wavelength (nm) used by
                           correction utilities.
    :type  ref_wavelength: float
    :param g: Scattering anisotropy factor
              (:math:`-1 \le g \le 1`).
    :type  g: float
    :param desc: Human-readable description for ``__str__``/``__repr__``.
    :type  desc: str
    :param display_color: Hex/CSS colour to use when plotting.
    :type  display_color: str | None

    :raises AssertionError: If *mu_s* or *mu_a* contains negative
        elements, or if *g* lies outside :math:`[-1, 1]`.

    **Computed attributes**

    :ivar mu_t: Total attenuation coefficient

        .. math::

            \mu_t(\lambda) \;=\; \mu_s(\lambda)\;+\;\mu_a(\lambda)

    :ivar albedo: Single-scattering albedo

        .. math::

            a(\lambda) \;=\; \frac{\mu_s(\lambda)}{\mu_t(\lambda)}

    **Notes**

    * Whenever :math:`\mu_s = 0`, the corresponding *g* value is forced
      to 1 and a warning is issued.
    * Array inputs can be queried at arbitrary wavelengths using the
      helper methods ``mu_s_at``, ``mu_a_at``, ``mu_t_at`` and
      ``albedo_at`` (linear interpolation).

    **Examples**

    .. code-block:: python

        dermis = Medium(
            n=1.4,
            mu_s=[15, 12, 10],          # at 600, 650, 700 nm
            mu_a=[0.30, 0.25, 0.20],
            wavelengths=np.array([600, 650, 700]),
            g=0.9,
            desc="dermis"
        )

        dermis.mu_t_at(650)             # 12.25 mm⁻¹
        dermis.albedo_at([600, 700])    # array([0.9804, 0.9804])
    """

    n: float = 1
    mu_s: float | np.ndarray = 0
    mu_a: float | np.ndarray = 0
    wavelengths: np.ndarray = wl
    ref_wavelength: float = 650
    g: float = 1
    desc: str = "default"
    display_color: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("mu_s", "mu_a", "wavelengths", mode="before")
    @classmethod
    def convert_to_array(cls, v: float | Iterable[float]) -> np.ndarray:
        """Ensure values (mu_s and mu_a) are non-negative."""
        assert np.all(v >= 0)
        return np.array(v)

    @field_validator("g", mode="before")
    @classmethod
    def g_vals(cls, g: float) -> float:
        """Ensure g is in range [-1, 1]"""
        assert -1 <= g <= 1
        return g

    @model_validator(mode="after")
    def force_non_scatterers_g(self) -> Medium:
        """Turn g to 1 if mu_s is 0 to simplify modelling. Send a warning to the user."""
        if np.any(self.mu_s == 0 and self.g != 1):
            warnings.warn(
                "g is automatically set to 1 where mu_s is 0. "
                "Changing `mu_s` will not reset `g`. You must reset it manually afer changing `mu_s`,"
                " or set a non-zero scattering coefficient if a non-unity g value is necessary."
            )
            self.g = np.where(self.mu_s == 0, 1, self.g)
        if iterable(self.g) and len(self.g) == 1:
            self.g = self.g[0]
        return self

    def __repr__(self):
        return self.desc.capitalize() + " Optical Medium Object"

    def __str__(self):
        if self.desc == "default":
            return f"Optical Medium: n={self.n}, mu_s={self.mu_s}, mu_a={self.mu_a}, g={self.g}"
        return self.desc.capitalize()

    def set(self, **kwargs: Any) -> None:
        """
        Sets multiple attributes of the object in bulk.

        This method is useful for updating the medium's attributes dynamically, particularly when modifying properties
        during iterative simulations while generating a lookup table. It assigns the provided keyword arguments to
        existing attributes of the object.

        :param kwargs: A dictionary of attribute names and their corresponding new values.
        :type kwargs: dict
        :return: None
        :raises AttributeError: If a provided attribute name does not exist.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _wave_index(self, wavelength: float | Iterable[float]):
        if iterable(wavelength) and iterable(self.wavelengths):
            return [np.where(self.wavelengths == wl)[0][0] for wl in wavelength]
        elif iterable(self.wavelengths):
            return np.where(self.wavelengths == wavelength)[0][0]
        else:
            return 0

    def mu_s_at(self, wavelengths: float):
        """Interpolate mu_s at the query wavelength, returns global mu_s if no function for mu_s."""
        if iterable(self.mu_s):
            return np.interp(wavelengths, self.wavelengths, self.mu_s)
        return self.mu_s

    def mu_a_at(self, wavelengths: float):
        """Interpolate mu_a at the query wavelength, returns global mu_s if no function for mu_a."""
        if iterable(self.mu_a):
            return np.interp(wavelengths, self.wavelengths, self.mu_a)
        return self.mu_a

    def mu_t_at(self, wavelengths: float):
        """Interpolate mu_t at the query wavelength, returns global mu_s if no function for mu_t."""
        if iterable(self.mu_t):
            return np.interp(wavelengths, self.wavelengths, self.mu_t)
        return self.mu_t

    def albedo_at(self, wavelengths: float):
        """Interpolate albedo at the query wavelength, returns global mu_s if no function for albedo."""
        if iterable(self.albedo):
            return np.interp(wavelengths, self.wavelengths, self.albedo)
        return self.albedo

    @property
    def mu_t(self):
        r"""Transport coefficient: :math:`\mu_t = \mu_s + \mu_a`"""
        return self.mu_s + self.mu_a

    @property
    def albedo(self):
        r"""Transpot albedo: :math:`a = \frac{\mu_a}{\mu_t}`"""
        if self.mu_t == 0:
            return 0
        else:
            return self.mu_a / self.mu_t


class Illumination:
    """
    Source model that *samples* photons from an angular/positional
    **pattern** and an optional **emission spectrum**.

    The class is lightweight: it simply stores callables that know
    how to generate photon starting conditions and delegates the heavy
    lifting to :class:`photon_canon.optics.Photon`.

    :param pattern: A callable ``pattern(batch_size) → (loc, dir)`` that
        returns

        * ``loc`` – Cartesian positions *(N × 3)* in millimetres.
        * ``dir`` – Directional cosines *(N × 3)*.

        If omitted, the default is a hollow-cone beam created by
        :py:func:`create_hollow_cone_beam`.
    :type  pattern: Callable[[int], Tuple[numpy.ndarray, numpy.ndarray]]
    :param spectrum: Iterable of *relative* intensities sampled at the
        global wavelength grid ``wl``.  If *None*, photons are
        initialised without a wavelength (i.e. monochromatic).
    :type  spectrum: Iterable[float] | None
    :param desc: Human-readable description used by :py:attr:`desc`.
    :type  desc: str | None

    :ivar pattern: The sampling function supplied at construction time.
    :ivar spectrum: The normalised emission spectrum (or *None*).
    :ivar description: Raw description string.

    .. rubric:: Examples

    >>> illum = Illumination(
    ...     pattern=create_hollow_cone_beam((0.2, 0.4), 35 * np.pi/180),
    ...     spectrum=my_led_spectrum,
    ...     desc="LED hollow cone"
    ... )
    >>> photon_packet = illum.photon(batch_size=1_000)
    >>> photon_packet.location.shape, photon_packet.direction.shape
    ((1000, 3), (1000, 3))
    """

    def __init__(
        self,
        pattern: Callable = create_hollow_cone_beam((ID, OD), THETA),
        spectrum: Optional[Iterable[Real]] = None,
        desc: Optional[str] = None,
    ) -> None:
        self.pattern = pattern
        self.spectrum = spectrum
        self.description = desc

    def photon(self, batch_size: int = 50000, **kwargs: Any) -> Photon:
        location, direction = self.pattern(batch_size)
        wavelength = (
            sample_spectrum(self.spectrum) if self.spectrum is not None else None
        )
        """Get a photon object sampled from the illumination function.
        
        :param batch_size: Number of photons to sample.
        :type batch_size: int
        :param kwargs: A dictionary of attribute names and their corresponding values to pass to the Photon init. 
        :return: A photon object with location and directions sampled from the illumination function.
        """
        return Photon(
            wavelength,
            batch_size=batch_size,
            location_coordinates=location,
            directional_cosines=direction,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"IlluminationObject(pattern={self.pattern}, spectrum={self.spectrum})"

    @property
    def desc(self) -> str:
        """A convenience description string for easier logging of simulations."""
        if self.description is None:
            return self.__repr__()
        return self.description


class Detector:
    """
    Simple tally detector that counts photons accepted by a user-supplied
    **acceptance function**.

    The detector maintains two running counters:

    * :py:attr:`n_total`    – total weight of photons presented.
    * :py:attr:`n_detected` – weight of photons whose exit positions and
      directions satisfy the acceptance criterion.

    :param acceptor: Callable that decides if a photon hits the detector.
        It is called as ``acceptor(x, y, mu_z=mu_z)`` where

        * ``x, y`` – exit coordinates of each photon (mm),
        * ``mu_z``  – := ``cos(θ_exit)`` (z-component of direction).

        Must return a Boolean array of the same length.
        The default is an aperture defined by
        :py:func:`create_cone_of_acceptance`.
    :type  acceptor: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
    :param desc: Optional description string.
    :type  desc: str | None

    :ivar n_total: Cumulative *weight* of all photons processed.
    :ivar n_detected: Cumulative *weight* of photons accepted.
    :ivar acceptor: Acceptance function provided at construction.
    :ivar description: Raw description string.

    :raises ValueError: If :py:func:`__call__` is invoked with an object
        that is not a :class:`~photon_canon.optics.Photon`.

    .. rubric:: Usage example

    >>> det = Detector(desc="f/2.4 collection cone")
    >>> det(photon_packet)          # tally all photons in the packet
    >>> det.n_detected / det.n_total   # detection efficiency
    0.031
    """

    def __init__(
        self,
        acceptor: Callable = create_cone_of_acceptance(ID),
        desc: Optional[str] = None,
    ) -> None:
        self.acceptor = acceptor
        self.n_total = 0
        self.n_detected = 0
        self.description = desc

    def detect(
        self,
        location: Iterable[float],
        direction: float | Iterable[float],
        weights: Optional[float | Iterable[float]] = None,
    ) -> None:
        """Determine if a photon is detected according to the detector function. Adds the weighted photon to the count
            if detected.

        :param location: Location of photon to detect.
        :type location: Iterable[float]
        :param direction: Direction of photon to detect.
        :type direction: float | Iterable[float]
        :param weights: Weights of photons to detect.
        :type weights: float | Iterable[float]
        """
        weights = weights if weights is not None else 1
        self.n_total += np.nansum(weights)
        x, y = location[:, :2].T
        mu_z = direction[:, -1]
        accepted_mask = self.acceptor(x, y, mu_z=mu_z)
        self.n_detected += np.nansum(weights[accepted_mask])

    def __call__(
        self, photon: Photon, mask: Optional[NDArray[bool]] = True
    ) -> None:
        """Callable interface for detection. See :py:meth:`detect`."""
        if not isinstance(photon, Photon):
            raise ValueError(
                "Detector object can only be called directly with a photon. "
                "Use detector.detect() for non-photon test cases."
            )
        self.detect(
            photon.exit_location[mask],
            photon.exit_direction[mask],
            photon.exit_weights[mask],
        )

    def reset(self) -> None:
        """Set detector counts back to 0"""
        self.n_total = 0
        self.n_detected = 0

    def __repr__(self) -> str:
        return f"Detector_object(acceptor=f{self.acceptor})"

    @property
    def desc(self) -> str:
        """A convenience description string for easier logging of simulations."""
        if self.description is None:
            return self.__repr__()
        return self.description


class System:
    def __init__(
        self,
        *args,
        surrounding_n: float = 1,
        illuminator: Optional[Illumination] = Illumination(pencil_beams),
        detector: Tuple[Detector, float] = (Detector(total_acceptor), 0)
    ) -> None:
        """
        Create a system of optical mediums and its surroundings that hold the optical properties and can determine the
        medium of a location as well as interfaces crossing given two locations. The blocks are constructed top down in
        the order they are received from the top down (positive z is downward) starting at 0 and surrounded by infinite
        surroundings.

        ### Process
        1. Create the surroundings using the input or default n
        2. Stack the surroundings from z=negative infinity to z=0
        3. Iterate through all *args, and create the following:
            - Dict of the system stack with Medium object keys and len=2 list of boundaries for respective
              object including surroundings
            - List of Medium object layers in order of stacking including surroundings
            - Ndarray of the boundary (i.e. interfaces) z location between Medium objects, excluding +- infinite
        4. Add surroundings to the bottom from z=system thickness to z= positive infinity if the last layer was not
        semi-infinite

        ### Paramaters
        :param *args: A variable number of ordered pairs of Medium objects and their respective thickness (in
        that order). The number of args input must be even.
        """
        self.illuminator = illuminator

        self.detector = detector[0]
        self.detector_location = detector[1]
        self._boundaries = [float("-inf"), 0, float("inf")]

        if not len(args) % 2 == 0:
            raise ValueError(
                "Arguments must be in groups of 2: medium object and thickness."
            )

        # Add surroundings from -inf to 0 to inf
        self.surroundings = Medium(n=surrounding_n, mu_s=0, mu_a=0, desc="surroundings")
        self.layer = [
            self.surroundings,
            self.surroundings,
        ]  # List of layers in order of addition

        # Iterate through args to stack layers
        for i in range(0, len(args), 2):
            self.add(args[i], args[i + 1])

    @property
    def boundaries(self) -> NDArray[float]:
        return np.asarray(self._boundaries)

    @boundaries.setter
    def boundaries(self, value: List):
        self._boundaries = value

    def __repr__(self) -> str:
        return (
            "|".join(
                [f" <-- {bound[0]} --> {layer}" for bound, layer in self.stack.items()]
            )
            + f" <-- {self.boundaries[-1]}|"
        )

    def __str__(self) -> str:
        stack = "\n"
        space = 0
        for bound, layer in self.stack.items():
            txt = layer.__str__()
            space = len(txt) if len(txt) > space else space
            txt = f"{bound[0]:.4g}"
            space = len(txt) if len(txt) > space else space
        txt = f"{bound[1]:.4g}"
        space = len(txt) if len(txt) > space else space
        space += 8

        for bound, layer in self.stack.items():
            txt = f" {bound[0]:.4g} "
            lfill = "-" * ((space - len(txt) - 2) // 2)
            rfill = "-" * (space - len(txt) - len(lfill) - 2)
            boundary = f"|{lfill}{txt}{rfill}|\n"

            txt = f" {layer} "
            lfill = " " * int(np.floor((space - len(txt) - 6) / 2))
            rfill = " " * (space - len(txt) - len(lfill) - 6)
            layer = f"|->{lfill}{txt}{rfill}<-|\n"
            stack += boundary + layer

        txt = f" {bound[1]:.4g} "
        lfill = "-" * ((space - len(txt) - 2) // 2)
        rfill = "-" * (space - len(txt) - len(lfill) - 2)
        boundary = f"|{lfill}{txt}{rfill}|\n"
        stack += boundary
        border = " " + "_" * (space - 2) + " "
        return border + stack + border

    @property
    def stack(self) -> dict[tuple[float, float], Medium]:
        """
        Provides an up-to-date dictionary representation of layer boundaries.

        This property dynamically constructs a dictionary mapping layer boundaries to their corresponding
        layers. It is useful for querying interfaces efficiently in float-time.

        :return: A dictionary where keys are tuples representing layer boundaries (lower_bound, upper_bound),
                 and values are the corresponding layers.
        :rtype: dict[tuple[float, float], Layer]
        """
        return {
            (self.boundaries[i], self.boundaries[i + 1]): layer
            for i, layer in enumerate(self.layer)
        }

    def add(self, layer: Medium, depth: float) -> None:
        """
        Adds a new layer of a given depth to the system.

        This method inserts a new `Medium` layer at the specified depth. If the preceding layer was
        semi-infinite, an error is raised. The method automatically replaces the surrounding filler layer
        before adding the new layer and ensures that the system remains properly bounded, extending back
        to infinity if necessary.

        :param layer: The `Medium` object representing the new layer.
        :type layer: Medium
        :param depth: The thickness of the new layer.
        :type depth: float
        :raises ValueError: If the preceding layer is semi-infinite.
        :return: None
        """

        if self.layer[-1] is not self.surroundings and self.boundaries[-1] == float(
            "inf"
        ):
            raise OverflowError("Cannot add to semi-infinite system.")
        depth = float(depth)

        # Get the bound of the last layer before surroundings
        replacement_idx = -2 if self.boundaries[-1] == float("inf") else -1
        start_interface = self.boundaries[replacement_idx]
        old_boundaries = self.boundaries[:replacement_idx]
        end_interface = start_interface + depth
        if replacement_idx == -2:
            self.layer[-1] = layer
        else:
            self.layer.append(layer)
        self.boundaries = np.append(old_boundaries, [start_interface, end_interface])

        # Add surroundings if not semi-infinte
        if end_interface < float("inf"):
            self.add(self.surroundings, float("inf"))

    def beam(self, batch_size: int = 50000, **kwargs: Any) -> Photon:
        """Gets a batch of photons from the integrated Illumination object.
        :param batch_size: The number of photons to return.
        :type batch_size: int
        :param kwargs: Additional keyword arguments to pass to the `Photon` constructor.
        :return: A `Photon` object sampled from the system's illumination.
        :rtype: Photon
        """
        photon = self.illuminator.photon(batch_size=batch_size, system=self)
        for key, val in kwargs.items():
            setattr(photon, key, val)
        return photon

    def in_medium(
        self, location: Iterable[float]
    ) -> NDArray[Medium | Tuple[Medium, Medium]]:
        """
        Return the medium(s) that are at the queried coordinate. If the coordinate is an interfaces location, the
        mediums that makeup the interfaces are returned as a tuple, this includes boundary interfaces being returned
        with False on the surroundings side.

        ### Process
        1. Get the z-coordinate from the input
        2. Check it against boundaries of each medium of the system until it is found that either:

            * It is between any of the boundaries, it is in the medium within those boundaries
            * It is at a boundary, it is "in" the two mediums that make up that interfaces

        3. Break and return the medium(s) of the queried point.

        :param location: The coordinates or z-coordinate to query.
        :type location: tuple, list, ndarray, float, or int
        :return `in_`: The medium of the queried z-coordinate or a tuple of interfaces if the coordinate is at an interfaces
        :rtype `in_`: medium | tuple of medium
        """
        location = np.asarray(location)
        z = location if np.ndim(location) == 1 else np.asarray(location)[:, 2]
        in_medium = np.empty_like(z, dtype=object)
        in_medium = np.where(z == float("inf"), self.layer[-1], in_medium)
        in_medium = np.where(z == float("-inf"), self.layer[0], in_medium)

        for bound, medium in self.stack.items():
            # If between boundaries, get that medium
            mask_inside = np.logical_and(bound[0] < z, z < bound[1])
            mask_boundary = (bound[0] == z) & (~np.isinf(z))

            in_medium[mask_inside] = medium

            # If at a boundary, get the mediums on each side
            if np.any(mask_boundary):
                z_neg_move = np.nextafter(z[mask_boundary], float("-inf"))
                z_pos_move = np.nextafter(z[mask_boundary], float("inf"))
                output1 = self.in_medium(z_neg_move)
                output2 = self.in_medium(z_pos_move)
                for i, idx in enumerate(np.where(mask_boundary)[0]):
                    in_medium[idx] = (output1[i], output2[i])

            # If all have been filled in, break
            if not np.any(np.equal(in_medium, None)):
                break
        return in_medium

    def interface_crossed(
        self, location0: Iterable[float], location1: Iterable[float]
    ) -> Tuple[Medium | (), float | None]:
        """Determines the first interfaces crossed when moving between two locations, considering only the z-coordinates.

        This method checks if any interfaces boundaries lie between the given z-coordinates. If an interfaces is crossed,
        the method calculates its z-location and identifies the two mediums forming the interfaces.

        ### Process:
        1. Identify boundaries that fall between the start and end z-coordinates.
        2. Compute the distance from the start z-coordinate to each boundary.
        3. Apply a mask to filter boundaries that are actually crossed.
        4. Select the closest crossed boundary as the interfaces plane.
        5. Determine the two mediums making up the interfaces by slightly shifting the plane's z-coordinate:

            * Backwards (toward the start) to find the first medium.
            * Forwards (away from the start) to find the second medium.

        :param location1: Starting location of the query.
        :param location2: Ending location of the query.
        :return: The two media forming the crossed interfaces, or an empty list `[]` if no interfaces is crossed and
         plane at which the interface is crossed or None.
        :rtype interfaces: tuple[Medium | (), float | None])
        """

        # Get z coords
        z0 = (
            np.asarray(location0)
            if np.shape(location0)[0] == 1
            else np.asarray(location0)[:, 2]
        )
        z1 = (
            np.asarray(location1)
            if np.shape(location1)[0] == 1
            else np.asarray(location1)[:, 2]
        )

        # Move off of boundaries (uncrossed!)
        is_boundary1 = np.isin(z0, self.boundaries)
        is_boundary2 = np.isin(z1, self.boundaries)

        z0 = np.where(is_boundary1, np.nextafter(z0, z1), z0)
        z1 = np.where(is_boundary2, np.nextafter(z1, z0), z1)

        # Sort for easier logic
        z_sorted = np.sort(np.stack((z0, z1), axis=-1), axis=-1)

        # Check if any boundaries fall between the zs and put into nd boolean array to use as a mask
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        crossed_mask = (z_sorted[..., 0, np.newaxis] < boundaries) & (
            boundaries < z_sorted[..., 1, np.newaxis]
        )

        # Determine closest crossed boundary (if any)
        dist_from_start = np.abs(boundaries - z0[..., None])
        dist_from_start[~crossed_mask] = np.inf  # Ignore non-crossed boundaries

        closest_idx = np.argmin(dist_from_start, axis=-1)

        plane = np.where(
            np.any(crossed_mask, axis=-1), boundaries[closest_idx], None
        ).astype(np.float64)

        # Determine mediums at the interface
        has_interface = np.any(crossed_mask, axis=-1)
        if np.any(has_interface):
            interface0 = np.where(
                has_interface, self.in_medium(np.nextafter(plane, z0)), None
            )
            interface1 = np.where(
                has_interface, self.in_medium(np.nextafter(plane, z1)), None
            )

            # Combine mediums into tuples or return empty tuples where no interface was crossed
            interfaces = np.array(
                [
                    tuple(pair) if valid else ()
                    for valid, pair in zip(has_interface, zip(interface0, interface1))
                ],
                dtype=object,
            )
            return interfaces, plane
        else:
            return None, None

    def represent_on_axis(self, ax: plt.axes.Axes = None) -> None:
        if ax is None:
            ax = plt.gca()
            lim = (
                [-0.1 * self.boundaries[-1], 1.1 * self.boundaries[-1]]
                if self.boundaries[-1] != float("inf")
                else [-0.1 * ax.get_ylim()[0], 1.1 * ax.get_ylim()[1]]
            )
            if ax.name == "3d":
                ax.set(zlim=lim)
        else:
            alpha = 0
            for bound, medium in self.stack.items():
                depth = np.diff(bound)
                y_edge = bound[0] - 0.1 * depth
                x_edge = ax.get_xlim()[0] * 0.95
                ax.text(x_edge, y_edge, medium.desc, fontsize=12)
                line_x = 100 * np.asarray(ax.get_xlim())
                y1 = bound[0] if not np.isinf(bound[0]) else ax.get_ylim()[0]
                y2 = bound[1] if not np.isinf(bound[1]) else ax.get_ylim()[1]
                alpha += 0.2
                ax.fill_between(
                    line_x,
                    y1,
                    y2,
                    color=(
                        "gray" if medium.display_color is None else medium.display_color
                    ),
                    alpha=alpha if medium.display_color is None else 1,
                )


class IndexableProperty(np.ndarray):
    """NumPy ``ndarray`` subclass that **renormalises itself** whenever its
    values change.

    The object behaves like a normal array but keeps an internal flag
    (:py:attr:`normalize`) that, when *True*, forces the vector(s) to be
    scaled to unit length after every assignment.

    :param arr: Initial data used to construct the array.
    :type  arr: Iterable
    :param normalize: If *True*, the array is normalised to unit length
                      on creation **and** after every in-place update.
    :type  normalize: bool, default ``False``
    :param dtype: NumPy data type to cast *arr* to.  If *None*, the dtype
                  is inferred by :py:func:`numpy.asarray`.
    :type  dtype: numpy.dtype | None
    :returns: A view of ``arr`` as an :class:`IndexableProperty`.
    :rtype: IndexableProperty

    :ivar bool normalize: Read/write property controlling automatic
        normalisation.  Setting

        >>> vec.normalize = True

        retroactively rescales the current data.

    **Notes**

    * Normalisation is performed along the **last axis**:

      .. math::

         \mathbf{x} \leftarrow
         \frac{\mathbf{x}}{\lVert \mathbf{x}\rVert_2}

    * When slicing, the returned view disables auto-normalisation so that
      scalar assignments do not accidentally rescale the parent array.

    **Examples**

    .. code-block:: python

        v = IndexableProperty([3, 4], normalize=True)
        float(v.norm())          # 1.0

        v[0] = 0                 # assignment triggers renormalisation
        list(v)                  # [0.0, 1.0]

        row = v[:]               # slice → normal ndarray behaviour
        row[1] = 10
        list(v)                  # still [0.0, 1.0]
    """

    def __new__(
        cls, arr: Iterable, normalize: bool = False, dtype: Optional[np.dtype] = None
    ) -> IndexableProperty:
        obj = np.asarray(arr, dtype=dtype).view(cls)
        obj.normalize = normalize
        obj /= np.linalg.norm(obj, axis=-1)[..., np.newaxis] if normalize else 1
        return obj

    def __array_finalize__(self, obj: Optional[NDArray]):
        if obj is None:
            return None
        self._normalize = getattr(obj, "_normalize", False)

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        self._normalize = value
        if self._normalize:
            self /= np.linalg.norm(self, axis=-1)[..., np.newaxis]

    def __setitem__(self, index: int, value: float):
        super().__setitem__(index, value)
        if self.normalize:
            self /= np.linalg.norm(self, axis=-1)[..., np.newaxis]

    def __getitem__(self, index: int) -> IndexableProperty[float]:
        item = super().__getitem__(index)
        if isinstance(item, IndexableProperty):
            item.normalize = False
        return item


# TODO: Make this work with 1 photon the same it does for batches
class Photon:
    """Simulated photon (or **batch** of photons) for Monte-Carlo light-transport modelling.

     The object tracks each photon’s wavelength, position, direction and statistical weight while it propagates through
      an optical :py:class:`~photon_canon.optics.System`. Built-in routines handle movement, scattering, absorption,
      interface interactions and Russian-roulette termination.

     :ivar int n: Number of photons in the batch.
     :ivar float | Iterable[float] wavelength: Wavelength(s) in nanometres.
     :ivar ~photon_canon.optics.System system: Optical system that contains the batch.
     :ivar IndexableProperty directional_cosines: Direction cosines *(μ_x, μ_y, μ_z)* for every photon.
     :ivar numpy.ndarray location_coordinates: Cartesian coordinates *(x, y, z)* for every photon.
     :ivar numpy.ndarray weights: Statistical weights (initially 1.0).
     :ivar float russian_roulette_constant: Survival threshold for Russian-roulette.
     :ivar bool recurse: Enable creation of secondary photons.
     :ivar int recursion_depth: Current recursion depth.
     :ivar int recursion_limit: Maximum permitted recursion depth.
     :ivar bool throw_recursion_error: Raise an error (``True``) or just warn (``False``) when the limit is exceeded.
     :ivar bool keep_secondary_photons: Retain secondary photons once spawned.
     :ivar float tir_limit: Cosine threshold for total-internal reflection.
     :ivar float A: Cumulative absorbed weight.
     :ivar float R: Cumulative reflected weight (back-exit).
     :ivar float T: Cumulative transmitted weight (forward-exit).
     :ivar numpy.ndarray exit_location: Terminal coordinates of each photon.
     :ivar numpy.ndarray exit_direction: Terminal direction cosines.
     :ivar numpy.ndarray exit_weights: Terminal weights.
     :ivar numpy.ndarray location_history: Complete trajectory history *(steps × n × 3)*.
     :ivar numpy.ndarray weights_history: Corresponding weight history *(steps × n)*.
     :ivar numpy.ndarray cache_register: Boolean mask—medium lookup cached?
     :ivar numpy.ndarray at_interface: Boolean mask—photon currently at an interface?

     **Key methods**

     * :py:meth:`__init__` – construct a photon packet.
     * :py:meth:`simulate` – propagate until all photons terminate (moves,
         absorbs, scatters, handles interfaces).
     * :py:meth:`absorb` – deposit weight according to medium albedo.
     * :py:meth:`move` – advance photons, handling boundary crossings.
     * :py:meth:`reflect_refract` – deterministic update at interfaces with Fresnel/TIR handling.
     * :py:meth:`scatter` – random direction change using phase function.
     * :py:meth:`russian_roulette` – survival test for low-weight photons.
     * :py:meth:`copy` – deep copy of the photon packet.
     * :py:meth:`plot_path` – visualise trajectories in 3-D or projection.

    """

    def __init__(
        self,
        wavelength: float | Iterable[float],
        batch_size: int = 0,
        system: Optional[System] = None,
        directional_cosines: Iterable[float] = (0, 0, 1),
        location_coordinates: Iterable[float] = (0, 0, 0),
        weights: float | Iterable[float] = 1,
        russian_roulette_constant: float = 20,
        recurse: bool = True,
        recursion_depth: Optional[int] = 0,
        recursion_limit: Optional[float] = 100,
        throw_recursion_error: bool = True,
        keep_secondary_photons: bool = False,
        tir_limit: Optional[float] = float("inf"),
    ) -> None:
        """Construct a photon batch and prime all run-time trackers.

        :param wavelength: Single wavelength or an iterable (nm).
        :type  wavelength: float | Iterable[float]
        :param batch_size: Number of photons *N* in the batch.
        :type  batch_size: int
        :param system: Optical :class:`~photon_canon.optics.System`
                       that will contain this photon.  May be set
                       later but must be non-``None`` before
                       :py:meth:`simulate` is called.
        :type  system: System | None
        :param directional_cosines: Either one triplet *(μₓ, μᵧ, μ_z)*
                                    replicated across the batch or an
                                    *(N × 3)* array.
        :type  directional_cosines: Iterable[float] | Iterable[Iterable[float]]
        :param location_coordinates: Initial positions *(x, y, z)* (mm).
        :type  location_coordinates: Iterable[float] | Iterable[Iterable[float]]
        :param weights: Initial statistical weights (defaults to 1.0).
        :type  weights: float | Iterable[float]
        :param russian_roulette_constant: Multiplier applied to surviving
                                          photons in the roulette step.
        :type  russian_roulette_constant: float
        :param recurse: Spawn secondary photons for reflected portions?
        :type  recurse: bool
        :param recursion_depth: **Internal use** – depth counter.
        :type  recursion_depth: int
        :param recursion_limit: Maximum recursion depth before abort.
        :type  recursion_limit: int
        :param throw_recursion_error: Raise or just warn when the limit
                                      is exceeded.
        :type  throw_recursion_error: bool
        :param keep_secondary_photons: Keep secondary packets in
                                       :py:attr:`secondary_photons`.
        :type  keep_secondary_photons: bool
        :param tir_limit: Maximum number of consecutive TIR events
                          before forced termination.
        :type  tir_limit: float

        :raises RecursionError: If *recursion_depth* ≥ *recursion_limit*
                                **and** *throw_recursion_error* is ``True``.
        :raises ValueError: If array-shaped inputs are incompatible with
                            *batch_size*.
        """
        # Init photon state
        self.batch_size = batch_size
        self.wavelength = wavelength
        self.system = system
        self.russian_roulette_constant = russian_roulette_constant
        self._medium = None
        self.recurse = recurse
        self.recursion_depth = recursion_depth
        self.recursion_limit = recursion_limit
        self.throw_recursion_error = throw_recursion_error
        self.keep_secondary_photons = keep_secondary_photons
        self.tir_limit = tir_limit

        # Setup batched attributes
        self._directional_cosines = IndexableProperty(
            self._batch_fill(directional_cosines, dtype=np.float64)
        )
        self._location_coordinates = self._batch_fill(
            location_coordinates, dtype=np.float64
        )
        self._weights = self._batch_fill(weights)

        # Init all-false cache register (so medium will be filled in for all)
        self.cache_register = np.repeat(np.False_, self.batch_size, axis=0)

        # Init empty medium cache and at_interface cache
        self._medium = np.empty((self.batch_size,), dtype=object)
        self.at_interface = np.empty((self.batch_size,), dtype=bool)

        # Exit trackers
        self.exit_direction = np.empty_like(self.directional_cosines)
        self.exit_direction[:] = np.nan
        self.exit_location = np.empty_like(self.location_coordinates)
        self.exit_location[:] = np.nan
        self.exit_weights = np.empty(self.batch_size)
        self.exit_weights[:] = np.nan

        self.secondary_photons = []

        # Call setter in case current_photon is DOA
        self.weights = weights

        # Init trackers
        self.location_history = self.location_coordinates[..., np.newaxis].copy()
        self.weights_history = self.weights[..., np.newaxis].copy()
        self.recursed_photons = np.zeros(self.batch_size, dtype=np.uint16)
        self.tir_count = np.zeros(self.batch_size, dtype=np.uint16)
        self.A = 0.0
        self.T = 0.0
        self.R = 0.0

    def _batch_fill(
        self, value: Iterable | NDArray, dtype: Optional[np.dtype] = None
    ) -> NDArray:
        """
        Broadcast *value* to shape ``(batch_size, …)``.

        :param value: Scalar or 1-D / 2-D array to expand.
        :type  value: Iterable | numpy.ndarray
        :param dtype: Optional dtype to cast to.
        :type  dtype: numpy.dtype | None
        :return: Broadcast array.
        :rtype: numpy.ndarray
        :raises ValueError: If the input shape is incompatible.
        """
        value = np.asarray(value, dtype=dtype)
        # Return already batched inputs
        if value.shape and value.shape[0] == self.batch_size:
            return value

        # Determine singlet size of value (in cases of pre-dimed/unsqueezed arrays, shape[-1] will give singlet shape)
        base_shape = 1 if np.ndim(value) == 0 else np.shape(value)[-1]
        if np.ndim(value) <= 1:
            value = np.repeat(value[np.newaxis, ...], self.batch_size, axis=0)
        elif np.ndim(value) == 2 and np.shape(value)[0] == 1:
            value = np.repeat(value, self.batch_size, axis=0)
        else:
            raise ValueError(
                f"Input is incompatible with batch size. Input must be shape ({base_shape},) or "
                f"({self.batch_size}, {base_shape}) but the input is {np.shape(value)}"
            )
        return value

    @property
    def location_coordinates(self) -> NDArray[float]:
        """
        Retrieves the current value of the photon location coordinates.

        :return: The current value of the location coordinates as an np.ndarray.
        :rtype: NDArray[float]
        """
        return self._location_coordinates

    @location_coordinates.setter
    def location_coordinates(self, location_coordinates: Iterable[float]) -> None:
        """
        Sets the current value of the location coordinates and update the medium cache register for those that have
        change.
        :param location_coordinates:
        :return:
        """

        # Ensure ndarray and fill to batch size
        location_coordinates = self._batch_fill(
            np.asarray(location_coordinates, dtype=np.float64)
        )

        # Determine change status
        unchanged = location_coordinates[:, 2] == self.location_coordinates[:, 2]

        # Set coordinates
        self._location_coordinates = location_coordinates

        # Update cache register
        self.cache_register = np.where(unchanged, self.cache_register, np.False_)

    @property
    def directional_cosines(self) -> IndexableProperty[float]:
        """
        Retrieves the current value of the photon directional cosines.

        :return: The current value of the directional cosines, wrapped in an `IndexableProperty` object.
        :rtype: IndexableProperty[float]
        """
        return self._directional_cosines

    @directional_cosines.setter
    def directional_cosines(self, directional_cosines: Iterable[float]) -> None:
        """
        Setter for the photon directional cosines that ensures the normalization is maintained and updated the cache
        register where the sign of the z-direction changes to trigger medium updates for those photons.

        This setter automatically re-assigns the updated value to an indexable property object,
        which handles normalization both at creation and when setting values using an indexed `__setitem__`.

        :param value: An iterable of float numbers representing the directional cosines of the photon.
        :type value: Iterable[float]
        :return: This method updates the directional cosines and does not return any value.
        :rtype: None
        """
        # Ensure Indexable Property and fill to batch size
        directional_cosines = IndexableProperty(
            self._batch_fill(directional_cosines, dtype=np.float64), normalize=True
        )

        # Determine changed directions
        unchanged = np.sign(directional_cosines[..., 2]) == np.sign(
            self.directional_cosines[:, 2]
        )

        # Set directional cosines
        self._directional_cosines = directional_cosines

        # Update cache register
        self.cache_register = np.where(unchanged, self.cache_register, np.False_)

    def copy(self, **kwargs: Any) -> Photon:
        """
        Creates a deep copy of the Photon object and allows for overwriting specific attributes using kwargs.

        This method creates a new instance of the Photon object with the same attributes as the original. It can also
        overwrite the values of specified attributes using the keyword arguments passed to it. The tracker attributes
        (`T`, `R`, `A`, `tir_count`, `recursed_photons`) are automatically reset to 0 in all cases.

        :param kwargs: Keyword arguments representing attributes to overwrite in the copied object.
        :type kwargs: dict
        :return: A new Photon object that is a deep copy of the original -- except it points to the same system -- with
            overwritten attributes if provided.
        :rtype: Photon
        """

        new_obj = copy.deepcopy(self)
        new_obj.system = self.system
        # Check for kwarg overwrites
        for key, value in kwargs.items():
            if hasattr(new_obj, key):
                setattr(new_obj, key, value)
            elif hasattr(new_obj, f"_{key}"):
                setattr(new_obj, f"_{key}", value)

        # Reset tracker attributes
        for key in ["T", "R", "A"]:
            setattr(new_obj, key, 0)
        for key in ["tir_count", "recursed_photons"]:
            setattr(new_obj, key, np.zeros(self.batch_size, dtype=np.uint8))
        new_obj.location_history = self.location_coordinates[..., np.newaxis].copy()
        new_obj.weights_history = self.weights[..., np.newaxis].copy()

        return new_obj

    def __repr__(self) -> str:
        """
        Returns a string representation of the Photon object for quick debugging or analysis.

        This method generates a human-readable string that includes the object's attributes and their values,
        providing an easy way to inspect the state of the Photon.

        :return: A string representation of the Photon object.
        :rtype: str
        """
        out = ""
        for key, val in self.__dict__.items():
            out += f"{key.strip('_')}: {val}\n"
        return out

    def simulate(self) -> None:
        """
        Simulates the behavior of a batch of photons until all photons are terminated and updates the object's
        attributes accordingly.

        The simulation involves the following steps:
        - **Absorption**: Photons interact with the medium and may be absorbed. (See `self.absorb`).
        - **Movement**: Photons move within the medium. (See `self.move`).
        - **Scattering**: Photons may scatter as they travel through the medium. (See `self.scatter`).

        The exact behavior of these events depends on the optical properties of the `Medium` object the photon is in,
        which is determined through queries to the `System` that contains the photon, i.e., `self.system`.

        :return: This method runs the simulation and does not return any value.
        :rtype: None
        """

        if not self.system is not None:
            raise RuntimeError(
                "Photon must be in an Optical System object to simulate."
            )
        if self.recursion_depth >= self.recursion_limit:
            raise RecursionError(
                "Maximum photon recursion limit reached. Recursion depth limit can be increased with the "
                "recursion_limit attribute.\n"
                "To switch this error off and throw a warning instead, set throw_recursion_error to FALSE. This "
                "will simulate the photon to the limit without recursion, rather than throwing an error."
            )
        while not self.is_terminated:
            self.absorb()
            self.move()
            self.scatter()

    @property
    def weights(self) -> NDArray[float]:
        """
        Retrieves the current weights of the photons.
        Retrieves the current weights of the photons.

        :return: An array of float values representing the photon weights.
        :rtype: NDArray[float]
        """
        return self._weights

    @weights.setter
    def weights(self, weights: float | Iterable[float]) -> None:
        """
        Sets the photon weights and applies Russian roulette if the weights falls below the threshold of 0.005.

        If the weights is below 0.005, the `russian_roulette` method is called to determine whether the photon
        survives or is terminated.

        :param weights: The new weights value(s) to set. Can be a single float number or an iterable of float numbers.
        :type weights: float | Iterable[float]
        :return: This method updates the weights in place and does not return a value.
        :rtype: None
        """
        if isinstance(weights, Real) or np.shape(weights)[0] == 1:
            weights *= np.ones(self.batch_size)
        self._weights = np.where(weights > 0, weights, 0)
        rr_check = (0 < weights) & (weights < 0.005)
        if np.any(rr_check):
            self.russian_roulette(rr_check)

    def russian_roulette(self, mask: NDArray[bool]) -> None:
        """
        Determines photon survival using the Russian roulette technique.

        If a photon survives, its weights is increased by `self.russian_roulette_constant`.
        If it does not survive, its weights is set to `0`, effectively terminating it.

        :param mask: A boolean array where `True` indicates that the corresponding photon is
                     subject to the Russian roulette survival test.
        :type mask: NDArray[bool]
        :return: This method modifies photon weights in place and does not return a value.
        :rtype: None
        """
        survival = np.random.rand(np.count_nonzero(mask)) < (
            1 / self.russian_roulette_constant
        )
        self._weights[mask] = np.where(
            survival, self._weights[mask] * self.russian_roulette_constant, 0
        )

    @property
    def medium(self) -> NDArray[Medium]:
        """
        Determines the current medium for the photons. If the current medium is cached, it is returned from the cache,
        otherwise it is updated by querying the system with the location.

        If the photons are at an interface, the function returns the medium the photon is moving into,
        using `headed_into`. Otherwise, it returns the current medium.

        :return: An array of `Medium` objects representing the current medium for the photons.
        :rtype: NDArray[Medium]
        """
        # Fill in from the cache
        cached = self._medium
        self._medium = np.empty((self.batch_size,), dtype=object)
        self._medium[self.cache_register] = cached[self.cache_register]

        # Update changed photons
        self._medium[~self.cache_register] = self.system.in_medium(
            self.location_coordinates[~self.cache_register]
        )

        # Photons that are at an interface have a tuple returned from system query
        self.at_interface[~self.cache_register] = np.array(
            [isinstance(medium, (tuple, list)) for medium in self._medium]
        )[~self.cache_register]

        # Get the headed_into medium for those photons
        if np.any(self.at_interface):
            self._medium[self.at_interface] = self.headed_into(mediums=self._medium)[
                self.at_interface
            ]

        # Update the cache register to all-true
        self.cache_register[:] = np.True_

        return self._medium

    @property
    def is_terminated(self) -> bool:
        """
        Checks if there are still photons in the batch to simulate.

        This property returns `True` if all photons have been terminated (i.e., they have exited the simulation
        or met termination criteria). Otherwise, it returns `False`.

        :return: A boolean value indicating whether the simulation has completed for all photons.
        :rtype: bool
        """

        self._is_terminated = np.all(
            [
                medium is self.system.surroundings for medium in self.medium
            ]  # Outside the system
            | (self.weights <= 0.0)  # Fully absorbed
            | np.any(
                np.isinf(self.location_coordinates)
                | np.isnan(self.location_coordinates),
                axis=1,
            )  # At infinite
        )
        return self._is_terminated

    def headed_into(
        self, mediums: Optional[NDArray[Medium | Tuple[Medium, Medium]]] = None
    ) -> NDArray[Medium]:
        """
        Determines which medium a photon is headed into, particularly when at an interface.

        If `mediums` is provided, the function uses it instead of re-querying :py:meth:`system.in_mediums` to determine the
        current state. In cases where `mediums` contains tuples `(Medium, Medium)`, the function selects the appropriate
        medium based on direction:
        - The first element (index `0`) represents the negative-direction medium.
        - The second element (index `1`) represents the positive-direction medium.

        If `mediums` is not a tuple, the function simply returns the same medium.

        :param mediums: An optional array of mediums. Each element can be:
            - A `Medium` object, indicating a single medium.
            - A tuple `(Medium, Medium)`, representing a negative and positive medium pair at an interface.
            If not provided, the function queries :py:meth:`system.in_mediums` instead.
        :type mediums: NDArray[Medium | Tuple[Medium, Medium]], optional
        :return: An array of `Medium` objects representing the medium the photon is headed into.
        :rtype: NDArray[Medium]
        """

        mediums = (
            self.system.in_medium(self.location_coordinates)
            if mediums is None
            else mediums
        )
        in_mask = np.array([isinstance(medium, (tuple, list)) for medium in mediums])
        neg_medium = [
            medium[0] if in_ else np.nan for in_, medium in zip(in_mask, mediums)
        ]
        pos_medium = [
            medium[1] if in_ else np.nan for in_, medium in zip(in_mask, mediums)
        ]
        headed_into = np.where(
            ~in_mask,
            mediums,
            np.where(self.directional_cosines[:, 2] < 0, neg_medium, pos_medium),
        )

        return headed_into

    # TODO: Add fluorescence support (dont forget to consider quantum yield < 1)
    def absorb(self) -> None:
        """
        Decrements the weights of the photon batch according to the albedo of the current medium.

        This method simulates the absorption of photons in the current medium. The weights of each photon is
        reduced based on the albedo, which represents the fraction of light that is reflected. The remaining
        weights is used to continue the photon simulation.

        The specific absorption process depends on the optical properties of the medium the photon is currently in.
        """

        absorbed_weights = self.weights * np.array(
            [medium.albedo_at(self.wavelength) for medium in self.medium]
        )
        self.A += np.sum(absorbed_weights)
        self.weights = self.weights - absorbed_weights

    def move(self, step: float | Iterable[float] = None) -> None:
        r"""Moves the photon one step in its current direction.

        The size of the step can be either provided directly, or it can be determined by the mean free path of the
        current medium, which is calculated from the transport coefficient (mu_t). If the photon is in a medium with
        mu_t = 0 (i.e., no scattering or absorption), the photon will automatically advance to the next interface along
        its direction.

        If a step size is provided, the photon moves in the direction specified by its directional cosines until it
        either completes the step or hits an interface. If the full step would cross an interface, only the portion of
        the step before the interface is executed. At the interface, the photon is refracted and reflected according to
        the media properties, and its weights is updated (decremented) based on the interaction.

        :param step: The distance the photon should move. If not provided, the step size is sampled based on the mean
         free path, :math:`-\frac{\ln\xi}{\mu_t}`
        :type  step: float | Iterable[float], optional
        :raises ValueError: If *step* is an array of incompatible shape.
        :return: None. The photon's location coordinates and directions are updated.
        """
        # Get current state
        mu_t = np.array([medium.mu_t_at(self.wavelength) for medium in self.medium])
        dir_cos = self.directional_cosines
        loc = self.location_coordinates

        # If scattering occurs, step is sampled from the distribution
        if step is None:
            step = np.where(
                mu_t > 0, -np.log(np.random.rand(self.batch_size)) / mu_t, float("inf")
            )
            step = np.where(self.weights > 0, step, 0)
        step = self._batch_fill(step)
        new_loc = loc + step[:, np.newaxis] * dir_cos

        # Determine which photons cross an interfaces
        interface, plane = self.system.interface_crossed(loc, new_loc)
        crossed = (~np.isnan(plane)) if plane is not None else False
        move_to_interface = False
        if np.any(crossed):
            interface_steps = np.where(
                crossed, (plane - loc[:, 2]) / dir_cos[:, 2], float("inf")
            )

            # Find photons that should move to the interfaces instead
            move_to_interface = (interface_steps < step) & crossed
            new_loc[move_to_interface] = (
                loc[move_to_interface]
                + interface_steps[move_to_interface, np.newaxis]
                * dir_cos[move_to_interface]
            )

        # Update location
        self.location_coordinates = new_loc

        # Reflect/refract photons at interfaces
        if np.any(move_to_interface):
            self.reflect_refract(interface, move_to_interface)

        # Update history for all photons
        self.location_history = np.append(
            self.location_history, new_loc[..., np.newaxis], axis=2
        )
        self.weights_history = np.append(
            self.weights_history, self.weights[..., np.newaxis], axis=1
        )

        # Check if any new photons exited
        exit_mask = np.array(
            [medium is self.system.surroundings for medium in self.headed_into()]
        ) & (self.weights > 0)

        if np.any(exit_mask):
            self.exit_location[exit_mask] = self.location_history[exit_mask, ..., -1]
            self.exit_weights[exit_mask] = self.weights_history[exit_mask, -1]

            # Check if any exited photons hit a detector
            detector_mask = exit_mask & (
                self.exit_location[:, 2] == self.system.detector_location
            )
            if np.any(detector_mask):
                self.system.detector(self, exit_mask)

            # Handle reflection or transmission
            self.R += np.sum(
                np.where(
                    exit_mask & (self.directional_cosines[:, 2] < 0), self.weights, 0
                )
            )
            self.T += np.sum(
                np.where(
                    exit_mask & (self.directional_cosines[:, 2] > 0), self.weights, 0
                )
            )

            # Terminate exited photons
            self.weights[exit_mask] = 0

    def reflect_refract(
        self,
        interfaces: NDArray[object[Tuple[Medium, Medium], Tuple[None]]],
        mask: NDArray[bool],
    ) -> None:
        """
        Computes photon reflection and refraction at interfaces.

        This method determines the weight of photon reflection and updates the tracker's direction accordingly.
        If reflection occurs, it either updates the photon direction or spawns secondary photons with adjusted weights
        to continue the recursive simulation. The photon's direction is updated based on the interface's refractive
        values. This operation is performed only on photons specified by the input mask.

        :param interfaces: NumPy array containing interface definitions.
                           Each element is a tuple of (Medium, Medium) for a valid interface or (None,) if no interface
                           is present.
        :type interfaces: np.ndarray[object]
        :param mask: Boolean mask indicating which photons to process.
        :type mask: NDArray[bool]
        :return: None. The photon's direction and weight are updated along with relevant trackers,
                 and secondary photons are spawned if necessary.
        """

        # Get incidence state
        mu_x, mu_y, mu_z_i = self.directional_cosines[mask].T
        mu_z_t = np.zeros_like(mu_z_i)
        n1 = np.array(
            [
                (
                    interface[0].n
                    if iterable(interface) and len(interface) == 2
                    else np.nan
                )
                for interface in interfaces
            ]
        )[mask]
        n2 = np.array(
            [
                (
                    interface[1].n
                    if iterable(interface) and len(interface) == 2
                    else np.nan
                )
                for interface in interfaces
            ]
        )[mask]

        # Calculate refraction
        sin_theta_t = n1 / n2 * np.sqrt(1 - (mu_z_i**2))

        # TIR
        tir_mask = sin_theta_t > 1
        mu_z_t[tir_mask] = -mu_z_i[tir_mask]
        self.tir_count[mask] = np.where(
            tir_mask, self.tir_count[mask] + 1, self.tir_count[mask]
        )
        stop_tir = self.tir_count[mask] > self.tir_limit
        self.A += np.sum(np.where(stop_tir, self.weights[mask], 0))
        self.weights[mask] = np.where(stop_tir, 0, self.weights[mask])

        # Snell's + Fresnel's Law
        refract_mask = ~tir_mask
        mu_z_t_masked = np.sqrt(1 - (sin_theta_t[refract_mask] ** 2))

        # Extract only the masked refractive values for masked values
        n1_masked = n1[refract_mask]
        n2_masked = n2[refract_mask]
        abs_mu_z_i = np.abs(mu_z_i[refract_mask])

        rs = (
            np.abs(
                ((n1_masked * abs_mu_z_i) - (n2_masked * mu_z_t_masked))
                / ((n1_masked * abs_mu_z_i) + (n2_masked * mu_z_t_masked))
            )
            ** 2
        )
        rp = (
            np.abs(
                ((n2_masked * mu_z_t_masked) - (n1_masked * abs_mu_z_i))
                / ((n1_masked * abs_mu_z_i) + (n2_masked * mu_z_t_masked))
            )
            ** 2
        )
        specular_reflection = 0.5 * (rs + rp) * self.weights[mask][refract_mask]

        mu_z_t[refract_mask] = mu_z_t_masked * np.sign(
            mu_z_i[refract_mask]
        )  # Ensure correct sign

        # Updated for transmitted portion
        mu_x[refract_mask] *= n1_masked / n2_masked
        mu_y[refract_mask] *= n1_masked / n2_masked

        if self.recurse:
            # Setup new photon attributes
            batch_size = np.sum(refract_mask).item()
            dir_cos = (
                np.array([[1, 1, -1]]) * self.directional_cosines[mask][refract_mask]
            )  # Flipped for reflection
            weights = specular_reflection
            loc_cor = self.location_coordinates[mask][refract_mask]
            rec_dep = self.recursion_depth + 1
            secondary_photons = Photon(
                self.wavelength,
                system=self.system,
                batch_size=batch_size,
                location_coordinates=loc_cor,
                directional_cosines=dir_cos,
                weights=weights,
                recursion_depth=rec_dep,
            )
            secondary_photons._medium = self._medium[mask][refract_mask]
            secondary_photons.cache_register = self.cache_register[mask][refract_mask]
            secondary_photons.at_interface = self.at_interface[mask][refract_mask]
            try:
                secondary_photons.simulate()
            except RecursionError as e:
                if self.throw_recursion_error:
                    raise e
                else:
                    warnings.warn(str(e), RuntimeWarning)
            finally:
                self.R += secondary_photons.R
                self.T += secondary_photons.T
                self.A += secondary_photons.A
                if self.keep_secondary_photons:
                    self.secondary_photons.append(secondary_photons)
                    self.secondary_photons += secondary_photons.secondary_photons

        else:
            # If the reflected fraction will be reflected out, add it to reflected count, Else add it to transmitted
            reflected_out = mu_z_i[refract_mask] > 0
            transmitted_out = mu_z_i[refract_mask] < 0
            self.R += np.sum(specular_reflection * reflected_out)
            self.T += np.sum(specular_reflection * transmitted_out)

        w_temp = self.weights[mask].copy()
        w_temp[refract_mask] -= specular_reflection
        self.weights[mask] = w_temp

        # Send to setter for normalization
        self.directional_cosines[mask] = np.column_stack((mu_x, mu_y, mu_z_t))

    def scatter(
        self,
        theta_phi: Optional[
            Iterable[float, float] | Iterable[Iterable[float, float]]
        ] = None,
    ):
        """
        This method updates the direction of the photon members where they are in scattering media, but not at an
        interface.

        :param theta_phi: Angles to update direction with. Optional.
        :type theta_phi: Iterable[float, float] | Iterable[Iterable[float, float]]
        :return: None. Updates photon directions where the photon is in scattering media (but not at an interface)
        """
        # Early break if all are at an interface or in non-scattering medium
        g = np.array(
            [medium.g for medium in self.medium]
        )  # (also forces reset of interface cache where necessary)
        if np.all(self.at_interface) or np.all(g == 1):
            return

        # Placeholders for angle samples
        theta = np.zeros(self.batch_size, dtype=np.float64)
        cosine_theta = np.zeros_like(theta, dtype=np.float64)
        if theta_phi is None:
            # Sample random scattering angles from distribution
            [xi, zeta] = np.random.rand(self.batch_size, 2).T

            # For non-zero g
            non_zero_g_mask = g != 0
            lead_coeff = 1 / (2 * g[non_zero_g_mask])
            term_2 = g[non_zero_g_mask] ** 2
            term_3 = (1 - term_2) / (
                1 - g[non_zero_g_mask] + (2 * g[non_zero_g_mask] * xi[non_zero_g_mask])
            )
            cosine_theta[non_zero_g_mask] = lead_coeff * (1 + term_2 - term_3)

            # For g=0
            cosine_theta[~non_zero_g_mask] = 1 - (2 * xi[~non_zero_g_mask])

            theta = np.arccos(cosine_theta)
            phi = 2 * np.pi * zeta
        else:
            theta, phi = (
                theta_phi if isinstance(theta_phi, (tuple, list)) else zip(theta_phi)
            )
        # Update direction cosines
        mu_x, mu_y, mu_z = self.directional_cosines.T
        new_directional_cosines = np.zeros((self.batch_size, 3), dtype=np.float64)

        # For near-vertical photons (simplify for stability)
        vertical = np.abs(mu_z) > 0.999
        new_directional_cosines[vertical, 0] = np.sin(theta[vertical]) * np.cos(
            phi[vertical]
        )
        new_directional_cosines[vertical, 1] = (
            np.sign(mu_z[vertical]) * np.sin(theta[vertical]) * np.sin(phi[vertical])
        )
        new_directional_cosines[vertical, 2] = np.sign(mu_z[vertical]) * np.cos(
            theta[vertical]
        )

        # For all others
        nonvertical = ~vertical
        deno = np.sqrt(1 - (mu_z[nonvertical] ** 2))

        # mu_x updated
        numr = (mu_x[nonvertical] * mu_z[nonvertical] * np.cos(phi[nonvertical])) - (
            mu_y[nonvertical] * np.sin(phi[nonvertical])
        )

        new_directional_cosines[nonvertical, 0] = (
            np.sin(theta[nonvertical]) * (numr / deno)
        ) + (mu_x[nonvertical] * np.cos(theta[nonvertical]))

        # mu_y update
        numr = (mu_y[nonvertical] * mu_z[nonvertical] * np.cos(phi[nonvertical])) + (
            mu_x[nonvertical] * np.sin(phi[nonvertical])
        )

        new_directional_cosines[nonvertical, 1] = (
            np.sin(theta[nonvertical]) * (numr / deno)
        ) + (mu_y[nonvertical] * np.cos(theta[nonvertical]))

        # mu_z update
        new_directional_cosines[nonvertical, 2] = -(
            np.sin(theta[nonvertical]) * np.cos(phi[nonvertical]) * deno
        ) + (mu_z[nonvertical] * np.cos(theta[nonvertical]))

        # Update directional cosines with new direction (done at once for normalization consistency)
        self.directional_cosines[~self.at_interface] = new_directional_cosines[
            ~self.at_interface
        ]

    def plot_path(
        self,
        project_onto: Optional[str] = None,
        axes: Optional[plt.Axes] = None,
        ignore_outside: bool = True,
    ) -> None:
        """
        Visualizes the photon location histories as a 2D or 3D plot.

        - If `project_onto` is specified, the path is projected onto a 2D plane.
          Accepted values: 'xy', 'xz', 'yz'.
        - If `project_onto` is None, a 3D plot is generated.
        - If `axes` is provided, the plot is drawn on the given `matplotlib.Axes` object.
          Otherwise, a new figure and axes are created.
        - If `ignore_outside` is True (default), only locations within the system are plotted,
          and steps after exit are truncated.

        :param project_onto: Plane onto which the path is projected ('xy', 'xz', 'yz'), or None for 3D.
        :type project_onto: Optional[str]
        :param axes: Matplotlib Axes object for plotting, or None to create new axes.
        :type axes: Optional[plt.Axes]
        :param ignore_outside: Whether to ignore steps after exiting the system (default: True).
        :type ignore_outside: bool
        :return: None
        """

        project_onto = ["xz", "yz", "xy"] if project_onto == "all" else project_onto
        project_onto = (
            [project_onto]
            if isinstance(project_onto, str) or project_onto is None
            else project_onto
        )
        batch_size, _, steps = (
            self.location_history.shape
        )  # Expect shape (batch, 3, steps)

        # Boundaries for filtering
        z_min, z_max = self.system.boundaries[0], self.system.boundaries[-1]
        inside = (
            (
                (self.location_history[:, 2] >= z_min)
                & (self.location_history[:, 2] <= z_max)
            )
            if ignore_outside
            else True
        )

        fig = (
            plt.figure(figsize=(8 * len(project_onto), 8))
            if not plt.get_fignums()
            else plt.gcf()
        )
        if project_onto[0]:
            axes = (
                [
                    fig.add_subplot(1, len(project_onto), i + 1)
                    for i in range(len(project_onto))
                ]
                if axes is None
                else axes
            )
            for ax, projection in zip(axes, project_onto):
                for i in range(batch_size):
                    x, y = (
                        self.location_history[i, "xyz".index(projection[0])],
                        self.location_history[i, "xyz".index(projection[1])],
                    )
                    ax.plot(x[inside[i]], y[inside[i]], label=f"Photon {i + 1}")
                ax.set_title(f"Projected onto {projection}-plane")
                ax.set_xlabel(f"Photon Displacement in {projection[0]}-direction (cm)")
                ax.set_ylabel(f"Photon Displacement in {projection[1]}-direction (cm)")
                if projection[1] == "z" and not ax.yaxis_inverted():
                    ax.invert_yaxis()
                if projection[0] == "z" and not ax.xaxis_inverted():
                    ax.invert_xaxis()
        else:
            axes = fig.add_subplot(projection="3d") if axes is None else axes
            for i in range(batch_size):
                x, y, z = (
                    self.location_history[i, 0],
                    self.location_history[i, 1],
                    self.location_history[i, 2],
                )
                axes.plot(
                    x[inside[i]], y[inside[i]], z[inside[i]], label=f"Photon {i + 1}"
                )
            axes.set_title("Photon Paths")
            axes.set_xlabel("Photon Displacement in x-direction (cm)")
            axes.set_ylabel("Photon Displacement in y-direction (cm)")
            axes.set_zlabel("Photon Displacement in z-direction (cm)")
            if not axes.zaxis_inverted():
                axes.invert_zaxis()

        return fig, axes
