from __future__ import annotations

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("simulation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.info("Starting the simulation script...")

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
    from Project.classes.State import StateMachine

    logger.info("Successfully loaded all required libraries.")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    raise ImportError(f"Failed to import required libraries: {e}") from e


def get_args_dict() -> dict:
    """
    Parse command-line arguments provided by the user.

    This function defines the following arguments:
    - `--initial-state` (`-i`): str, required
        Path to the initial state file (.yaml) that defines the
        simulation's starting conditions.
    - `--trajectories` (`-t`): int, optional, default=5
        Number of trajectories for the simulation.
    - `--steps` (`-s`): int, optional, default=100
        Number of steps to run in each trajectory.
    - `--output-folder` (`-o`): str, optional, default="output"
        Path to the folder where the simulation's
        output files (.npy and .png) will be saved.

    Returns
    ------
    dict:
        A dictionary containing parsed arguments with argument
        names as keys and user-provided values.
    """
    description = "run gillespie simulation"

    parser = ArgumentParser(description=description)

    parser.add_argument(
        "-i",
        "--initial-state",
        dest="initial_state",
        type=str,
        required=False,
        help="defines path to initial state file (.yaml)",
    )

    parser.add_argument(
        "-t",
        "--trajectories",
        dest="trajectories",
        type=int,
        required=False,
        default=5,
        help="defines number of trajectories",
    )

    parser.add_argument(
        "-s",
        "--steps",
        dest="steps",
        type=int,
        required=False,
        default=100,
        help="defines number of steps",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        dest="output_folder",
        type=str,
        required=False,
        default="output",
        help="""defines path to output folder
            (save .npy and .png simulation plots)""",
    )

    args_dict = vars(parser.parse_args())
    return args_dict


def main() -> None:
    """
    Coordinates the execution of the simulation.

    Based on user input in command-line.

    This function performs the following:
    - Parses arguments using `get_args_dict`.
    - Initializes the initial state using the
        `State` and `StateMachine` classes.
    - Runs the simulation with parameters for
        the number of trajectories and steps.
    - Plots and saves the simulation results if an output folder is specified.

    Workflow
    --------
    1. Parse arguments to retrieve:
        - `input_path`: Path to the initial state file.
        - `output_folder`: Path where outputs will be saved.
        - `save`:
            Boolean indicating whether to save outputs based on `output_folder`
        - `trajectories`: Number of trajectories to simulate.
        - `steps`: Number of steps in each trajectory.
    2. Run simulation using `State` and `StateMachine` classes.
    3. Plot and save the simulation output.

    Returns
    -------
    None
    """
    logger.info("Starting the simulation...")

    # parsing args
    args_dict = get_args_dict()
    logger.info(f"Parsed arguments: {args_dict}")

    input_path = args_dict["initial_state"]
    output_folder = args_dict["output_folder"]
    logger.info(f"Input path: {input_path}, Output folder: {output_folder}")

    save = output_folder is not None
    trajectories = args_dict["trajectories"]
    steps = args_dict["steps"]

    # Input validation
    if steps <= 0:
        raise ValueError("Number of steps must be a positive integer.")
    if trajectories <= 0:
        raise ValueError("Number of trajectories must be a positive integer.")

    simulator = StateMachine(init_state_path=input_path)
    simulator.run(
        steps=steps,
        trajectories=trajectories,
        save_path=output_folder,
        save=save,
    )
    logger.info("Simulation completed successfully.")

    simulator.plot(save_folder=output_folder)
    logger.info("Plotting completed.")


if __name__ == "__main__":
    main()
