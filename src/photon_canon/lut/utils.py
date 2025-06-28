import sqlite3
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import (
    Union,
    Iterable,
    Optional,
    Callable,
    TypeVar,
    ParamSpec,
    Generic,
    Type,
)

from ..import_utils import np

from .. import System, Illumination
from ..utils import latest_simulation_id

"""
Photon-Canon – Monte-Carlo LUT helpers
=====================================

Database utilities for the lookup-table (LUT) subsystem:

* :pyfunc:`add_metadata` – insert one simulation-run header row  
* :pyfunc:`add_system_data` – persist the fixed-layer optical stack  
* :pyfunc:`add_simulation_result` – bulk-insert voxel-level results  
* :pyfunc:`_get_sim_id` – resolve a simulation-ID for class/instance calls
"""

# High level typing helpers
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

# Set up simulation database
db_dir = Path.home() / ".photon_canon"
db_dir.mkdir(parents=True, exist_ok=True)
db_path = db_dir / "lut.db"
con = sqlite3.connect(db_path)
c = con.cursor()


class LUTError(Exception):
    pass


class Dimensions(str, Enum):
    """
    Enumeration of optical-property dimensions.

    :var MU_S: – scattering coefficient :math:`\mu_s`
    :var MU_A: – absorption coefficient :math:`\mu_a`
    :var G:    – Henyey–Greenstein anisotropy factor *g*
    """

    MU_S = "mu_s"
    MU_A = "mu_a"
    G = "g"


class ListPortion(Enum):
    """
    Handy constant slices for printing lists/arrays.

    :var HEAD: ``slice(None, 5)`` – first five items
    :var TAIL: ``slice(-5, None)`` – last five items
    :var ALL:  ``slice(None, None)`` – the entire list
    """

    HEAD = slice(None, 5)
    TAIL = slice(-5, None)
    ALL = slice(None, None)


class classOrInstanceMethod(Generic[P, R, T]):
    """
    Decorator that lets a function behave as **both** an instance *and* a
    class-method.

    The wrapped function receives two extra arguments::

        (instance, owner, *args, **kwargs)

    * *instance* – ``self`` when called on an instance, else ``None``
    * *owner*    – the owning class object

    :param func: Function to wrap.
    :type  func: Callable
    """

    def __init__(
        self, func: Callable[[Optional[T], Type[T], P.args, P.kwargs], R]
    ) -> None:
        self.func = func

    def __get__(self, instance: Optional[T], owner: Type[T]) -> Callable[P, R]:

        def func(*args: P.args, **kwargs: P.kwargs) -> R:
            return self.func(instance, owner, *args, **kwargs)

        return func


# --------------------------------------------------------------------------- #
#  DB insertion helpers                                                       #
# --------------------------------------------------------------------------- #
def add_metadata(
    n: int = None, recursive: bool = False, detector: Illumination = None
) -> int:
    """
    Insert a new row into the metadata table, **mclut_simulations**, and return its *id*.

    :param n: Number of photons simulated; ``None`` if not recorded.
    :type  n: int | None, *optional*
    :param recursive: ``True`` if the Monte-Carlo was run with recursive
        scattering, defaults to ``False``.
    :type  recursive: bool, *optional*
    :param detector: Optional detector object. If supplied, its ``.desc`` string
        is stored in *detector_description*.
    :type  detector: Any, *optional*
    :return: Auto-incremented primary-key of the inserted simulation.
    :rtype: int
    """

    # Parse to get metadata
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS mclut_simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photon_count INTEGER NOT NULL,
            recursive BOOLEAN DEFAULT FALSE,
            detector BOOLEAN DEFAULT FALSE,
            detector_description TEXT DEFAULT ''
        );
        """
    )

    # Insert row
    c.execute(
        """
        INSERT INTO mclut_simulations
            (photon_count, recursive, detector, detector_description)
        VALUES (?, ?, ?, ?)
        """,
        (
            n,
            recursive,
            detector is not None,
            detector.desc if detector is not None else "",
        ),
    )
    con.commit()

    return c.lastrowid


def add_system_data(simulation_id: int, system: System) -> None:
    """
    Persist the optical-system stack for a given *simulation_id*.

    One row per layer is written to **fixed_layers**; thickness and optical
    properties are frozen at the layer’s reference wavelength.

    :param simulation_id: Foreign-key referencing *mclut_simulations.id*.
    :type  simulation_id: int
    :param system: The :py:class:`~photon_canon.System` whose layers are recorded.
    :type  system: System
    """
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS fixed_layers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stack_order INTEGER NOT NULL,
            layer TEXT NOT NULL,
            mu_s REAL NOT NULL,
            mu_a REAL NOT NULL,
            g REAL NOT NULL,
            thickness REAL NOT NULL,
            ref_wavelength REAL NOT NULL,
            simulation_id INTEGER NOT NULL,
            FOREIGN KEY (simulation_id) REFERENCES mclut_simulations(id)
        );
        """
    )

    # Generate fixed layer details for table
    fixed_layers = []
    for i, (bound, layer) in enumerate(system.stack.items()):
        fixed_layers.append(
            (
                int(i),
                layer.desc,
                float(layer.mu_s_at(layer.ref_wavelength)),
                float(layer.mu_a_at(layer.ref_wavelength)),
                float(layer.g),
                float(bound[1] - bound[0]),
                float(layer.ref_wavelength),
                int(simulation_id),
            )
        )

    # Bulk-insert
    c.executemany(
        """
        INSERT INTO fixed_layers
            (stack_order, layer, mu_s, mu_a, g, thickness, ref_wavelength, simulation_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        fixed_layers,
    )
    con.commit()


def add_simulation_result(
    simulation_id: int,
    mu_s: Union[Real, Iterable[Real]],
    mu_a: Union[Real, Iterable[Real]],
    g: Union[Real, Iterable[Real]],
    depth: Union[Real, Iterable[Real]],
    output: Union[Real, Iterable[Real]],
) -> None:
    """
    Bulk-insert Monte-Carlo voxel results into **mclut**. Bulk insert protects against partial LUT addition, and limits
     I/O time, however, this can easily be looped to write single rows at a time.

    Scalars are broadcast to match iterable inputs; all iterable arguments must
    share the same length.

    :param simulation_id: Simulation to which the rows belong.
    :type  simulation_id: int
    :param mu_s: Scattering coefficient(s) :math:`\mu_s`.
    :type  mu_s: Real | Iterable[Real]
    :param mu_a: Absorption coefficient(s) :math:`\mu_a`.
    :type  mu_a: Real | Iterable[Real]
    :param g: Anisotropy factor(s) *g*.
    :type  g: Real | Iterable[Real]
    :param depth: Depth coordinate(s) where *output* was recorded.
    :type  depth: Real | Iterable[Real]
    :param output: Recorded quantity (e.g. reflectance).
    :type  output: Real | Iterable[Real]
    :raises ValueError: If iterable arguments have mismatched lengths.
    :return: ``None`` – rows are committed to the database.
    :rtype: None
    """

    # Ensure all are iterable arays
    arrays = [np.array(val) for val in [mu_s, mu_a, g, depth, output, simulation_id]]
    iters = [arr.ndim == 1 for arr in arrays]
    shapes = [arrays[idx].shape for idx, itr in enumerate(iters) if itr]

    # Check that sizes are compatible
    if shapes and not np.all(shapes == shapes[0]):
        raise ValueError("Iterables must have the same shape")
    elif not shapes:
        shapes = [[1]]

    # Expand non-iterables to match and make all floats
    for i, itr in enumerate(iters):
        if not itr:
            arrays[i] = np.repeat([float(arrays[i])], shapes[0][0])
        else:
            arrays[i] = np.array([float(val) for val in arrays[i]])

    # Put into tuples for executemany
    data_tuples = tuple(zip(*arrays))

    # Add table (if not exists)
    c.execute(
        """
         CREATE TABLE IF NOT EXISTS mclut (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             mu_s REAL NOT NULL,
             mu_a REAL NOT NULL,
             g REAL NOT NULL,
             depth REAL NOT NULL,
             output REAL NOT NULL,
             simulation_id INTEGER NOT NULL,
             FOREIGN KEY (simulation_id) REFERENCES mclut_simulations(id)
         );
         """
    )

    # Add results to db
    c.executemany(
        f"""
        INSERT INTO mclut 
            (mu_s, mu_a, g, depth, output, simulation_id) 
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        data_tuples,
    )

    con.commit()


def _get_sim_id(
    obj: Optional["LUT"], simulation_id: int | None, set_default: bool = True
) -> int | None:
    """
    Resolve which *simulation_id* to use inside class/instance helper methods.

    Order of precedence
    -------------------
    #. If *obj* is **not** ``None`` (instance call):
       * Use *simulation_id* only if it matches ``obj.simulation_id``.
       * Otherwise raise :py:class:`~LUTError`.
    #. If *obj* is ``None`` (class call) and *simulation_id* is ``None``:
       * Return :py:data:`~photon_canon.utils.latest_simulation_id`
         when *set_default* is ``True``.
    #. Otherwise return the explicit *simulation_id* (may be ``None``).

    :param obj: LUT instance to get the simulation ID from or None when coming form a class method call.
    :type obj: LUT | None
    :param simulation_id: Simulation ID to set/check or None if not input.
    :type simulation_id: int | None
    :param set_default: Whether the simulation ID is set to a default value or not.
    :type set_default: bool
    :return: A simulation ID or None if not set_default.
    :rtype: int | None
    """
    if obj is not None:
        if simulation_id is not None and simulation_id != obj.simulation_id:
            raise LUTError(
                f"Input simulation_id ({simulation_id})does not match simulation_id of instance ({obj.simulation_id}). "
                "Consider calling as class method with simulation_id argument instead."
            )
        return simulation_id or obj.simulation_id

    elif simulation_id is None and set_default:
        return latest_simulation_id

    else:
        return simulation_id
