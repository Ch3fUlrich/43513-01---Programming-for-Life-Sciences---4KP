# defines auxiliary functions to main module

from __future__ import annotations

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("simulation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

try:
    from pathlib import Path
    import copy
    from typing import List, Dict, Optional, Union
    from argparse import ArgumentParser
    import numpy as np
    from numba import njit, prange
    import matplotlib.pyplot as plt
    from tqdm import trange
    import yaml

    logger.info("Successfully loaded all required libraries.")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    raise ImportError(f"Failed to import required libraries: {e}") from e

@njit(parallel=True)
def fast_random_occurrence(
    expression_rate: float,
    from_count: int,
) -> np.ndarray:
    """
    Simulate random occurrences for molecules at a given expression rate.

    This function is 4 times faster than np.random.choice.

    Parameters
    ----------
    expression_rate: float
        The probability rate at which a molecule is expressed,
        typically between 0 and 1.
    from_count: int
        The number of molecules

    Returns
    -------
    int
        The total number of molecules that are expressed.
    """

    # Precompute the cumulative probability
    cumulative_prob = 1 - expression_rate
    occurrences = np.zeros(from_count, dtype=np.int32)

    # Loop through each element and assign based on random value
    for i in prange(from_count):
        rand_val = np.random.rand()  # Generate a random float in [0, 1)
        if rand_val > cumulative_prob:
            occurrences[i] = 1

    num_occurrences = np.sum(occurrences)
    return num_occurrences


def construct_path(path: Optional[str] = None, fname: Optional[str] = None) -> Path:
    """
    Construct a file path for simulation states.
    If the path is not provided, the current working directory is used.

    Parameters
    ----------
    path : str, optional
        The directory path to the file. Defaults to the
        current working directory.
    fname : str, optional
        The name of the file. Defaults to "init_state.yaml".

    Returns
    -------
    Path
        The constructed file path.

    Raises
    ------
    ValueError
        If the constructed file path does not have a ".yaml" extension.
    """
    if not path:
        project_folder = Path(__file__).parent
        state_folder = "states"
        fname = "init_state.yaml"
        path = project_folder.joinpath(state_folder, fname)

    if Path(path).suffix != ".yaml":
        raise ValueError("File must be a yaml file")
    return path
