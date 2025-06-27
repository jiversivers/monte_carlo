import itertools
import logging
import multiprocessing as mp
import warnings
from numbers import Real
from typing import Optional, List, Union, Tuple, Type, Any, Callable

from pydantic import PrivateAttr, BaseModel, model_validator
from tqdm import tqdm

from .utils import LUTError, _get_sim_id
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


def _single_sim(
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
                        pool.imap(_single_sim, params),
                        total=len(params),
                        desc=f"Sim ID: {simulation_id}",
                    )
                )
            else:
                results = list(pool.imap(_single_sim, params))
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
    """
    Lookup table class that acts as an interface to generated LUT data. This class includes callability with one or
    multiple parameters, with the parameter call order set at LUT object creation by `dimensions`. When called, the LUT
    will automatically handle extrapolation based on its settings and return the response value accordingly. The LUT
    surface can be smoothed by passing a smoohting_fn, which is applied to the grid at interpolation time. The LUT also
    provides several methods to visualize or access the underlying LUT data. Finally, class/instance methods of the LUT
    class provide a simplified interface for querying the available information across multiple simulations.

    :param simulation_id: The ID of the generated LUT to access.
    :type simulation_id: int
    :param dimensions: The dimensions to consider at lookup in the LUT. The order will be used for calls. Default: 'mu_s', 'mu_a', 'g'.
    :type dimensions: List[str]
    :param extrapolate: Whether to extrapolate outside the table bounds. Overrideable at call time as a kwarg. Default: False.
    :type extrapolate: bool
    :param scale: The scale to compare responses against to normalize (if needed). Default: 1.
    :type scale: float
    :param smoothing_fn: A function that will be applied to the response grid. Optional. Default: None.
    :type smoothing_fn: callable
    """

    simulation_id: int = latest_simulation_id
    dimensions: List[Dimensions] = [Dimensions.MU_S, Dimensions.MU_A, Dimensions.G]
    extrapolate: bool = False
    scale: float = 1
    smoothing_fn: Optional[Callable[[NDArray], NDArray]] = None

    _interpolator: Optional[RegularGridInterpolator] = PrivateAttr(default=None)

    class Config:
        ignored_types = (classOrInstanceMethod,)
        use_enum_values = True
        arbitrary_types_allowed = True

    def __call__(
        self, *values: Union[Real, np.ndarray], extrapolate: bool = None
    ) -> Union[Real, np.ndarray]:
        """
        Callable interface that looks up the response value according to the inputs along the object's set dimensions.
        If only the number of arguments provided is fewer than the tables dimensions, all response points for that set
        of arguments will be returned.

        :param values: Query points of the LUT,order by the `dimensions` of the object.
        :param extrapolate: Whether to extrapolate for values outside the table's bounds.
        :return response: LUT response at the queried values.
        """
        if not isinstance(values, tuple):
            values = (values,)
        if not len(values) <= len(self.dimensions):
            raise LUTError(
                f"LUT supports only up to {len(self.dimensions)}D. {self.dimensions}"
            )

        # Ensure all inputs are numpy arrays
        pts = [np.atleast_1d(v) for v in values]

        # Ensure all input arrays have the same shape for element-wise pairing
        input_shapes = [p.shape for p in pts]
        if len(set(input_shapes)) > 1:
            raise LUTError(
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
        """
        :return: The interpolator to use for interpolation of the response.
        """
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

            # Apply smoothing
            if self.smoothing_fn is not None:
                output = self.smoothing_fn(output)

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
    def to_pandas(
        self: Optional["LUT"], cls: Type["LUT"], simulation_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        A class or instance method that gets a dataframe of the LUT data.
        :param simulation_id: The id of the simulation to get.
        :return: DataFrame of the LUT data.
        """
        simulation_id = _get_sim_id(self, simulation_id)
        query = f"SELECT mu_s, mu_a, g, output FROM mclut WHERE simulation_id = {simulation_id}"
        df = pd.read_sql_query(query, CON)
        if self.smoothing_fn is not None:
            warnings.warn(
                "Smoothing is not applied to output dataframe. Reshape and apply manually if desired.",
                stacklevel=2,
            )
        return df

    @classOrInstanceMethod
    def list_available(
        self: Optional["LUT"], cls: Type["LUT"], portion: str = "ALL"
    ) -> List[int]:
        """
        A class or instance method that gets a list of available LUT simulation IDs.
        :param portion: What portion of the available LUT table to get. Available options are 'HEAD', 'TAIL', and 'ALL'. Default: 'ALL'.
        :return: A lsit of the available LUTs simulation IDs.
        """
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
    def get_layer_data(
        self: Optional["LUT"], cls: Type["LUT"], simulation_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        A class or instance method that gets a dataframe of the LUT layer data.
        :param simulation_id: The id of the simulation to get data from. Optional. If None, all available layer dats is retrieved. Default: None.
        :return: A dataframe of the LUT layer data.
        """
        simulation_id = _get_sim_id(self, simulation_id)
        query = f"SELECT * FROM fixed_layers WHERE simulation_id = {simulation_id}"
        df = pd.read_sql_query(query, CON)
        return df

    @classOrInstanceMethod
    def get_metadata(
        self: Optional["LUT"],
        cls: Type["LUT"],
        simulation_id: Optional[int] = None,
        portion: str = "ALL",
    ) -> pd.DataFrame:
        """
        A class or instance method that gets a dataframe of the LUT simulation metadata. Note, this dataframe will not include smoothing, regardless of self.smoothing_fn. For smoothed LUT array outputs, see :ref:`surface`.

        :param simulation_id: The id of the simulation to get the metadata from. Optional. If None, all available metadata is retrieved, and returned portion depends on `portion`. Default: None.
        :param portion: What portion of the available metadata to return. Available options are 'HEAD', 'TAIL', and 'ALL'. Default: 'ALL'.
        :return: A dataframe of the LUT simulation metadata.
        """
        query = f"SELECT * FROM mclut_simulations"
        portion = ListPortion[portion.upper()].value
        simulation_id = _get_sim_id(self, simulation_id, set_default=False)
        if simulation_id is not None:
            query += f" WHERE id = {simulation_id}"
        df = pd.read_sql_query(query, CON)
        return df[portion]

    def surface(self) -> Tuple[Real, Real, Real]:
        """
        .. _surface:

        Get the surface information for the LUT. Returns all unique values of each of the first 2 dimensions and the grid of responses at those dimensions. This allows for quick and easy visualization using ax.plot_wireframe(X, Y, Z). Note, this surface will include smoothing if self.smoothing_fn is not None.

        :return: both independent variables and the dependent response surface.
        """
        df = self.to_pandas(self.simulation_id)
        x = df[self.dimensions[0]].unique()
        y = df[self.dimensions[1]].unique()
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.reshape(df["output"], (len(x), len(y)))
        Z /= self.scale
        if self.smoothing_fn is not None:
            Z = self.smoothing_fn(Z)
        return X, Y, Z
