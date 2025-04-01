import itertools
from numbers import Real
from typing import Optional, List, Union, Tuple

from ..import_utils import np, NDArray, RegularGridInterpolator
import pandas as pd

from ..optics import System, Medium
from ..optics import Photon

from ..lut.utils import add_metadata, add_system_data, add_simulation_result
from ..utils import latest_simulation_id, CON

c = CON.cursor()


def generate_lut(
        system: System,
        variable: Medium,
        arrays: dict[str, NDArray],
        photon: Photon,
        pbar: bool = False,
        output: Optional[str] = None,
        **kwargs
) -> int:
    """
    Simulates photon transport for different optical properties to generate a lookup table (LUT).

    This function iterates over the input arrays, modifying the optical properties of `variable`
    (a `Medium` object) accordingly. A set of `Photon` objects is then simulated within the `system`.

    :param output: Determines what output measure to add to the LUT. Default total reflectance.
    :param system: The optical system in which the photons are simulated.
    :type system: System
    :param variable: The medium whose properties are varied in the LUT generation.
    :type variable: Medium
    :param arrays: Dictionary mapping property names to NumPy arrays containing values to iterate over.
    :type arrays: dict[str, np.ndarray]
    :param photon: A reference `Photon` object used to initialize the simulated photons.
    :type photon: Photon
    :param pbar: Whether to show a progress bar.
    :type pbar: bool
    :param kwargs: Additional optional parameters for the simulation.
    :return: The ID of the generated LUT of simulation.
    :rtype: int
    """

    # Add LUT metadata to db
    simulation_id = add_metadata(n=photon.batch_size, recursive=photon.recurse, detector=system.detector)
    add_system_data(simulation_id, system)

    # Setup progress bar (if desired)
    iterable = itertools.product(*arrays.values())
    if pbar:
        from tqdm import tqdm
        iterable = tqdm(iterable, total=np.prod([len(arr) for arr in arrays.values()]))

    # Iterate through all permutations of iterables
    copy_of_photon = photon.copy()
    for values in iterable:
        keys_values = dict(zip(arrays.keys(), values))
        if pbar:
            iterable.set_description(
                f' - '.join(f"{k}: {v}" for k, v in keys_values.items())
            )
        variable.set(**keys_values)

        # Reset and simulate
        photon = copy_of_photon.copy()
        if system.detector is not None:
            system.detector.reset()
        photon.simulate()

        # Get the output
        if system.detector is not None:
            target = system.detector.n_detected
        elif output is not None:
            target = getattr(photon, output, None)
        else:
            target = getattr(photon, 'R', None)

        # Update LUT
        for bound, layer in system.stack.items():
            if layer == variable:
                depth = bound[1] - bound[0]
                break
        add_simulation_result(simulation_id, variable.mu_s, variable.mu_a, variable.g, depth, target)
    return simulation_id

class LUT:
    def __init__(self,
                 simulation_id: int = latest_simulation_id,
                 dimensions: List[str] = ['mu_s', 'mu_s', 'g'],
                 extrapolate: bool = False):
        self.simulation_id = simulation_id
        self.dimensions = dimensions
        self.extrapolate = extrapolate
        self._interpolator = None

    def __call__(self,
                 *values: Union[Real, Tuple[Real]],
                 extrapolate: bool = None) -> Union[Real, np.ndarray[Real]]:
        """Supports 1D, 2D, and 3D lookups."""
        if not isinstance(values, tuple):
            values = (values,)
        if not len(values) <= len(self.dimensions):
            raise IndexError(f"LUT supports only up to {len(self.dimensions)}D. {self.dimensions}")

        # Fill in db range for unqueried dimensions
        pts = []
        for i, dim in enumerate(self.dimensions):
            if i >= len(values):
                c.execute(f"""
                SELECT DISTINCT {dim} FROM mclut WHERE simulation_id={self.simulation_id}
                """)
                fetched = c.fetchall()
                pts.append(np.unique([row[0] for row in fetched]))
            else:
                pts.append([values[i]])

        # Find nearest bounds for interpolation. Here we assume cubic spline interpolation for all parameters.
        # Build ND mesh to interpolate within
        mesh = np.meshgrid(*pts, indexing='ij')
        query_pts = np.array([m.flatten() for m in mesh]).T

        # Set/Reset extrapolation option
        interpolator = self.interpolator
        interpolator.bounds_error = ~extrapolate if extrapolate is not None else self.extrapolate
        return interpolator(query_pts)

    @property
    def interpolator(self) -> RegularGridInterpolator:
        if self._interpolator is None:
            query = f"SELECT {', '.join(self.dimensions)}, output FROM mclut WHERE simulation_id={self.simulation_id}"
            c.execute(query)
            results = c.fetchall()
            if results is None or len(results) == 0:
                raise IOError(
                    f'No simulations found at id {self.simulation_id}. Run generate_lut '
                    f'and save results to lut.db before using lookup or try a '
                    f'different ID.')

            # If no exact match, check if parameters are within the bounds for interpolation
            *values, output = zip(*results)

            # Get data into a regular grid
            points = tuple(np.unique(val) for val in values)
            output = np.asarray(output).reshape(*[len(p) for p in points])

            # Interpolation/Extrapolate (if True)
            self._interpolator = RegularGridInterpolator(
                points, output, method='cubic', bounds_error=not self.extrapolate, fill_value=None
            )

        return self._interpolator

    @classmethod
    def to_pandas(cls, simulation_id=None):
        if simulation_id is None:
            if hasattr(cls, 'simulation_id'):
                simulation_id = cls.simulation_id
            else:
                simulation_id = latest_simulation_id
        query = f"SELECT mu_s, mu_a, g, output FROM mclut WHERE simulation_id = {simulation_id}"
        df = pd.read_sql_query(query, CON)
        return df

    def surface(self) -> Tuple[Real]:
        df = self.to_pandas()
        x = df['mu_s'].unique()
        y = df['mu_a'].unique()
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.reshape(df['output'], (len(x), len(y)))
        return X, Y, Z
