import itertools
import logging
import multiprocessing as mp
from numbers import Real
from typing import Optional, List, Union, Tuple

from pydantic import PrivateAttr, BaseModel, ConfigDict
from tqdm import tqdm

from ..import_utils import np, NDArray, RegularGridInterpolator
import pandas as pd

from ..optics import System, Medium
from ..optics import Photon

from ..lut.utils import (
    add_metadata,
    add_system_data,
    add_simulation_result,
    Dimensions,
    ListPortion,
    classOrInstanceMethod,
)
from ..utils import latest_simulation_id, CON

c = CON.cursor()


def single_sim(
    args: Tuple[
        System, Medium, Photon, list[str, ...], list[float, ...], Optional[str]
    ],
):
    """Worker function to simulate a single case"""
    system, variable, photon, keys, values, output = args
    keys_values = dict(zip(keys, values))
    variable.set(**keys_values)

    # Reset and simulate
    local_photon = photon.copy()
    if system.detector is not None:
        system.detector.reset()
    local_photon.simulate()

    # Get the output
    if system.detector is not None:
        target = system.detector.n_detected
    elif output is not None:
        target = getattr(photon, output, None)
    else:
        target = getattr(photon, "R", None)

    # Update LUT
    for bound, layer in system.stack.items():
        if layer == variable:
            depth = bound[1] - bound[0]
            break

    return variable.mu_s, variable.mu_a, variable.g, depth, target


def generate_lut(
    system: System,
    variable: Medium,
    arrays: dict[str, NDArray],
    photon: Photon,
    pbar: bool = False,
    output: Optional[str] = None,
    num_workers: int = None,
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
    :param num_workers: Number of parallel processes to simulate photons.
    :return: The ID of the generated LUT of simulation.
    :rtype: int
    """

    # Add LUT metadata to db
    simulation_id = add_metadata(
        n=photon.batch_size, recursive=photon.recurse, detector=system.detector
    )
    add_system_data(simulation_id, system)

    # Prepare the list of parameter combinations
    param_combinations = list(itertools.product(*arrays.values()))
    keys = list(arrays.keys())

    # Prepare arguments for multiprocessing
    params = [
        (system, variable, photon, keys, values, output)
        for values in param_combinations
    ]

    # Process through all permutations of iterables
    num_workers = num_workers or mp.cpu_count()
    with mp.Pool(processes=num_workers) as pool:
        try:
            if pbar:
                results = list(
                    tqdm(
                        pool.imap(single_sim, params),
                        total=len(params),
                        desc=f"Sim ID: {simulation_id}",
                    )
                )
            else:
                results = list(pool.imap(single_sim, params))
        except Exception as e:
            logging.debug(f"Multiprocessing failed: {e}")
            results = []
        finally:
            pool.close()
            pool.join()

    for result in results:
        add_simulation_result(simulation_id, *result)

    c.close()
    return simulation_id


class LUT(BaseModel):
    simulation_id: int = latest_simulation_id
    dimensions: List[Dimensions] = [Dimensions.MU_S, Dimensions.MU_A, Dimensions.G]
    extrapolate: bool = False
    scale: float = 1

    _interpolator: Optional[RegularGridInterpolator] = PrivateAttr(default=None)

    class Config:
        ignored_types = (classOrInstanceMethod,)
        use_enum_values = True
        arbitrary_types_allowed = True

    def __call__(
        self, *values: Union[Real, np.ndarray], extrapolate: bool = None
    ) -> Union[Real, np.ndarray]:
        if not isinstance(values, tuple):
            values = (values,)
        if not len(values) <= len(self.dimensions):
            raise IndexError(
                f"LUT supports only up to {len(self.dimensions)}D. {self.dimensions}"
            )

        # Ensure all inputs are numpy arrays
        pts = [np.atleast_1d(v) for v in values]

        # Ensure all input arrays have the same shape for element-wise pairing
        input_shapes = [p.shape for p in pts]
        if len(set(input_shapes)) > 1:
            raise ValueError(
                f"Input arrays must have the same shape for element-wise pairing, got {input_shapes}"
            )

        # For unqueried dimensions, get full range from the database
        num_input_dims = len(values)
        for i in range(num_input_dims, len(self.dimensions)):
            c.execute(
                f"SELECT DISTINCT {self.dimensions[i]} FROM mclut WHERE simulation_id={self.simulation_id}"
            )
            pts.append(np.unique([row[0] for row in c.fetchall()]))

        # Pair elements
        query_pts = np.column_stack(pts)

        # Interpolation
        interpolator = self.interpolator
        interpolator.bounds_error = (
            not extrapolate if extrapolate is not None else not self.extrapolate
        )
        result = interpolator(query_pts)

        # Return result with the same shape as input arrays
        return result.reshape(input_shapes[0]) / self.scale

    @property
    def interpolator(self) -> RegularGridInterpolator:
        if self._interpolator is None:
            query = f"SELECT {', '.join(self.dimensions)}, output FROM mclut WHERE simulation_id={self.simulation_id}"
            c.execute(query)
            results = c.fetchall()
            if results is None or len(results) == 0:
                raise IOError(
                    f"No simulations found at id {self.simulation_id}. Run generate_lut "
                    f"and save results to lut.db before using lookup or try a "
                    f"different ID."
                )

            # If no exact match, check if parameters are within the bounds for interpolation
            *values, output = zip(*results)

            # Get data into a regular grid
            points = tuple(np.unique(val) for val in values)
            output = np.asarray(output).reshape(*[len(p) for p in points])

            # Interpolation/Extrapolate (if True)
            self._interpolator = RegularGridInterpolator(
                points,
                output,
                method="cubic",
                bounds_error=not self.extrapolate,
                fill_value=None,
            )

        return self._interpolator

    @classOrInstanceMethod
    def to_pandas(self, cls, simulation_id=None) -> pd.DataFrame:
        if simulation_id is None:
            simulation_id = latest_simulation_id
        query = f"SELECT mu_s, mu_a, g, output FROM mclut WHERE simulation_id = {simulation_id}"
        df = pd.read_sql_query(query, CON)
        return df

    @classOrInstanceMethod
    def list_available(self, cls, portion: str = "ALL") -> List[int]:
        c.execute(
            """
        SELECT DISTINCT id FROM mclut_simulations
        """
        )
        portion = ListPortion[portion.upper()].value
        ids = c.fetchall()
        available = []
        for id in ids:
            c.execute(
                f"""
            SELECT COUNT(*) from mclut WHERE simulation_id={id[0]}
            """
            )
            if c.fetchone()[0] > 1:
                available.append(id[0])
        return available[portion]

    @classOrInstanceMethod
    def get_layer_data(self, cls, simulation_id: Optional[int] = None) -> pd.DataFrame:
        if simulation_id is None:
            if hasattr(cls, "simulation_id"):
                simulation_id = cls.simulation_id
            else:
                simulation_id = latest_simulation_id
        query = f"SELECT * FROM fixed_layers WHERE simulation_id = {simulation_id}"
        df = pd.read_sql_query(query, CON)
        return df

    @classOrInstanceMethod
    def get_metadata(
        self, cls, simulation_id: Optional[int] = None, portion: str = "ALL"
    ) -> pd.DataFrame:
        query = f"SELECT * FROM mclut_simulations"
        portion = ListPortion[portion.upper()].value
        if simulation_id is None:
            if hasattr(cls, "simulation_id"):
                simulation_id = cls.simulation_id
                query += f" WHERE id = {simulation_id}"
        df = pd.read_sql_query(query, CON)
        return df[portion]

    def surface(self) -> Tuple[Real, Real, Real]:
        df = self.to_pandas(self.simulation_id)
        x = df[self.dimensions[0]].unique()
        y = df[self.dimensions[1]].unique()
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.reshape(df["output"], (len(x), len(y)))
        if hasattr(self, "scale"):
            Z /= self.scale
        return X, Y, Z
