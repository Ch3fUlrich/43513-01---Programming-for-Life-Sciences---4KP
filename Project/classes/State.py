# defines State related classes

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
    from Project.classes.Molecule import Molecule, Complex
    from Project.utils import construct_path

    logger.info("Successfully loaded all required libraries.")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    raise ImportError(f"Failed to import required libraries: {e}") from e

class State:
    """
    Represents the state of a cell in the simulation.

    This class manages the simulation state, including molecule counts and
    their behaviors, by loading and updating a YAML file that describes the
    initial state.

    Attributes
    ----------
    path : pathlib.Path
        Path to the state YAML file.
    state_dict : dict
        Dictionary containing the cell's current state.
    time : int
        Current simulation time.
    molecules : dict
        Dictionary of molecule objects in the state.

    Methods
    -------
    load_state()
        Loads the initial state from the YAML file.
    set_init_state()
        Initializes the simulation state.
    save_state()
        Saves the current state to a YAML file.
    create_state_dict(t, save=False)
        Creates and updates the state as a dictionary.
    create_molecules()
        Creates molecule objects for the simulation.
    next_state(dt=1)
        Advances the simulation by one time step.
    extract_molecule_counts(as_dict=False)
        Extracts the molecule counts as a dictionary or NumPy array.
    print(short=False, full=False)
        Prints the current state in short, full, or default format.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the state from a YAML file.

        Parameters
        ----------
        path: str
            The path to the state YAML file. Must be a valid `.yaml` file.

        Raises
        ------
        ValueError
            If the provided path is not a valid YAML file.
        """
        logger.info(f"Initializing state from file: {path}")
        if not isinstance(path, str) and not isinstance(path, Path):
            logger.error("Path must be a string or Path object.")
            raise ValueError("Path must be a string or Path object")
        assert Path(path).suffix == ".yaml", "File must be a yaml file"

        self.path: Path = Path(path)
        self.state_dict: Dict[str, Union[str, int, float, List[int]]] = None
        self.time: int = None
        self.molecules: Dict[str, Union["Molecule", "Complex"]] = None
        self.set_init_state()
        logger.info("State initialization completed successfully.")

    def load_state(self) -> Dict[str, Union[str, int, float, List[int]]]:
        """
        Load the state of the cell from the specified YAML file.

        Returns
        -------
        dict:
            The state of the cell as a dictionary

        Raises
        ------
        ValueError
            If the provided `path` is not a string or a `Path` object, or if it
            do not point to a valid `.yaml` file.
        """
        if self.path:
            with open(self.path, "r", encoding="UTF-8") as file:
                state = yaml.safe_load(file)
        else:
            raise ValueError("Path to initial state is not defined")
        return state

    def set_init_state(self) -> None:
        """
        Initialize the state by loading the YAML file.

        Setting up the initial simulation state.
        This method loads the state dictionary from the YAML file using
        `load_state()` and initializes the simulation time and molecules
        based on the loaded data.

        Attributes
        ----------
        state_dict : dict
            The dictionary containing the full state information loaded
            from the YAML file.
        time : int
            The current simulation time, extracted from the
            loaded state dictionary.
        molecules : dict
            A dictionary of molecules in the state, created using the
            `create_molecules()` method.
            This contains molecule names as keys and their properties
            as values.

        Returns
        -------
        None
        """
        self.state_dict = self.load_state()
        self.time: int = self.state_dict["time"]
        self.molecules: Dict[str, Union[Molecule, Complex]] = self.create_molecules()

    def save_state(self) -> None:
        """
        Save the current state of the cell to a YAML file.

        This method writes the `state_dict` attribute, which
        contains the current
        state of the cell, to the specified YAML file defined in `self.path`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with open(self.path, "w", encoding="UTF-16") as file:
            yaml.dump(self.state_dict, file)

    def create_state_dict(
        self, t: int, save: bool = False
    ) -> Dict[str, Union[str, int, float, List[int]]]:
        """
        Create and updates the state of the cell as a dictionary.

        Method generates the current state of the cell (including the
        simulation time) and the states of all molecules.
        Optionally, if save set to True, state is saved to a YAML file.

        Parameters
        ----------
        t: int
            Current simulation time.
        save: bool
            If True, save the state to a YAML file

        Returns
        -------
        dict:
            The updated state dictionary.
        """
        state_dict: Dict[str, Union[str, int, float, List[int]]] = {"time": t}
        for _, molecule in self.molecules.items():
            molecule_dict = molecule.create_molecule_dict()
            state_dict.update(molecule_dict)
        if save:
            self.save_state()
        return self.state_dict

    def create_molecules(self) -> Dict[str, Union[Molecule, Complex]]:
        """
        Create molecule objects based on the initial state.

        Returns
        -------
        dict:
            A dictionary of molecule objects.
        """
        molecules = {}
        for name, molecule_dict in self.state_dict.items():
            if name == "time":
                continue

            if name == "complex":
                molecules[name] = Complex(name=name, **molecule_dict)
            else:
                molecules[name] = Molecule(name=name, **molecule_dict)
        return molecules

    def next_state(self, dt: int = 1) -> "State":
        """
        Update the simulation time by one time step.

        Calculates changes in molecule
        counts based on transcription, decay,
        translation, and complex formation and updates the state
        dictionary with the new molecule counts. The updated
        state object is returned.

        Parameters
        ----------
        dt : int, optional
            The time step for the simulation. Default is 1.

        Returns
        -------
        state:
            The updated state object after applying the changes
            for the next time step.
        """
        # 1. Update the time
        self.time = self.time + dt

        # 2. Calculate changes in time for each molecule
        m = copy.deepcopy(self.molecules)
        # Perform molecule updates
        for molecule_name, molecule in m.items():
            # Log the state of the molecule before updating
            logger.debug(f"Before update - {molecule_name}: {molecule.count}")

        # TF_mRNA
        TF_mRNA_trancribed = m["TF_mRNA"].transcription(dt)
        TF_mRNA_decayed = m["TF_mRNA"].decay(dt)
        TF_mRNA_translated = m["TF_mRNA"].translation(dt)

        # TF_protein
        TF_protein_decayed = m["TF_protein"].decay(dt)

        # miRNA
        miRNA_trancribed = m["miRNA"].transcription(m["TF_protein"].count, dt)
        miRNA_decayed = m["miRNA"].decay(dt)

        # mRNA
        mRNA_trancribed = m["mRNA"].transcription(m["TF_protein"].count, dt)
        mRNA_decayed = m["mRNA"].decay(dt)
        mRNA_translated = m["mRNA"].translation(dt)

        # protein
        protein_decayed = m["protein"].decay(dt)

        # complex
        available_miRNA = m["miRNA"].count - miRNA_decayed
        available_mRNA = m["mRNA"].count - mRNA_decayed
        formed_complex, used_molecules = m["complex"].formation(
            [available_miRNA, available_mRNA],
            dt,
        )
        complex_degraded = m["complex"].degradation(dt)

        # update molecule counts with safeguards
        m["TF_mRNA"].count = max(
            0, m["TF_mRNA"].count + TF_mRNA_trancribed - TF_mRNA_decayed
        )
        m["TF_protein"].count = max(
            0, m["TF_protein"].count + TF_mRNA_translated - TF_protein_decayed
        )
        m["miRNA"].count = max(
            0,
            m["miRNA"].count
            + miRNA_trancribed
            - miRNA_decayed
            - used_molecules[0]
            + complex_degraded,
        )
        m["mRNA"].count = max(
            0,
            m["mRNA"].count + mRNA_trancribed - mRNA_decayed - used_molecules[1],
        )
        m["protein"].count = m["protein"].count + mRNA_translated - protein_decayed
        m["complex"].count = m["complex"].count + formed_complex - complex_degraded

        for molecule_name, m_class in m.items():
            if m_class.count < 0:
                raise ValueError(f"Negative count for {molecule_name}")

        self.molecules = m
        self.create_state_dict(self.state_dict["time"], save=False)
        # self.print(short=True)
        return self

    def extract_molecule_counts(
        self, as_dict: bool = False
    ) -> Union[Dict[str, int], np.ndarray]:
        """
        Extract molecule counts as a dictionary or NumPy array.

        Retrieves the counts of all molecules in the current state
        and returns them either as a dictionary or as a NumPy array, depending
        on the value of the `as_dict` parameter.

        Parameters
        ----------
        as_dict : bool, optional
            If True, returns a dictionary where keys are molecule names and
            values are their counts. If False, returns a NumPy array of counts.
            Default is False.

        Returns
        -------
        Union[Dict[str, int], np.ndarray]
            If `as_dict=True`: A dictionary with molecule names as
            keys and counts as values.
            If `as_dict=False`: A NumPy array of molecule counts.
        """
        if as_dict:
            counts = {
                molecule_name: molecule.count
                for molecule_name, molecule in self.molecules.items()
            }
        else:
            counts = np.array(
                [molecule.count for _, molecule in self.molecules.items()],
                dtype=int,
            )
        return counts

    def print(self, short: bool = False, full: bool = False) -> str:
        """
        Print the current state of the cell.

        Three possible formats: short, full, or default.

        Parameters
        ----------
        short: bool
            If True, prints only the molecule counts. Default is False.
        full: bool
            If True, print the full details of state of the cell.
            Default is False.

        Returns
        -------
        None
        """
        current_time = self.time
        output = f"""------------------------\nState(t={current_time})\n"""
        if full:
            for name, key_values in self.state_dict.items():
                if isinstance(key_values, dict):
                    output += f"- {name}:\n"
                    for attribute, value in key_values.items():
                        output += f"   - {attribute}: {value}\n"
                else:
                    output += f"- {name}: {key_values}\n"
        elif short:
            output += f"{self.extract_molecule_counts()}"
        else:
            for molecule_name, molecule in self.molecules.items():
                output += f"-> {molecule_name}: {molecule.count}\n"
        print(output)
        return output


class StateMachine:
    """
    Represents the state machine that drives the simulation.

    This class manages the simulation workflow, including initializing states,
    running multiple trajectories, saving results, and visualizing outputs.

    Attributes
    ----------
    state : State
        The current state of the simulation.
    path : pathlib.Path
        Path to the directory containing state files.
    dt : int
        Time step for the simulation.
    times : np.ndarray
        Array of recorded time points for the simulation.
    molecule_counts : np.ndarray
        A 3D array storing molecule counts across time steps and trajectories.

    Methods
    -------
    save_runs(molecule_counts, fname=None, path=None)
        Saves the results of the simulation runs.
    reset()
        Resets the simulation state to its initial configuration.
    run(steps=100, trajectories=100, save=True,
        saven_fname=None, save_path=None)
        Runs the simulation for a specified number of steps and trajectories.
    plot(example=False, scale="linear", save_folder=None)
        Generates a plot of the simulation results.
    """

    def __init__(
        self,
        init_state_path: Optional[str] = None,
        state: Optional[State] = None,
        dt: int = 1,
    ):
        """
        Initialize a state machine for simulation.

        Parameters
        ----------
        innit_state_path : str, optional
            Path to the YAML file describing the initial state of the cell.
            Cannot be used together with `state`.
        state : State, optional
            A pre-initialized `State` object for the simulation. Cannot be used
            together with `innit_state_path`.
        dt : int, optional
            The time step for the simulation. Default is 1.

        Raises
        ------
        ValueError
            If both `innit_state_path` and `state` are provided.
        """
        if init_state_path is not None and state is not None:
            if state.path != init_state_path:
                raise ValueError(
                    f"""Provided initial state path and state object do not
                        have the same path.
                        init_state_path: {init_state_path}
                        state.path: {state.path}"""
                )
        init_state_path = construct_path(init_state_path)
        self.state: State = State(init_state_path)

        self.path: Path = self.state.path.parent
        self.dt: int = dt
        self.times: Optional[np.ndarray] = None
        self.molecule_counts: Optional[np.ndarray] = None

    def save_runs(
        self,
        molecule_counts: np.ndarray,
        fname: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        """
        Save the results of the simulation runs.

        Parameters
        ----------
        molecule_counts : np.ndarray
            Array of molecule counts from the simulation.
        fname : str, optional
            File name to save the results. Defaults to "molecule_counts.npy".
        path : str, optional
            Directory to save the file. Defaults to the simulation directory.
        """
        fname = fname or "molecule_counts"
        path = path or self.path
        fpath = Path(path).parent.joinpath("output", fname)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {fpath}")
        try:
            np.save(fpath, molecule_counts)
            logger.info("Results saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise MemoryError(f"Failed to save results: {e}") from e

    def reset(self) -> None:
        """
        Reset the state machine to its initial configuration.

        Resets the `state`, `times`, and `molecule_counts` attributes.

        Returns
        -------
        None
        """
        self.state = State(self.state.path)
        self.times = None
        self.molecule_counts = None

    def run(
        self,
        steps: int = 100,
        trajectories: int = 100,
        save: bool = True,
        saven_fname: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Run the state machine for multiple steps and trajectories.

        Parameters
        ----------
        steps : int, optional
            Number of time steps for each trajectory.
            Default is 100.
        trajectories : int (default is 100)
            Number of independent simulation trajectories to run.
            Default is 100.
        save : bool
            Whether to save the results. Default is True (=save results).
        saven_fname : str, optional
            File name for saving the results. If not provided, a default name
            in the format `molecule_counts_{steps}_{trajectories}.npy` is used.
        save_path : str, optional
            Directory to save the results.

        Returns
        -------
        numpy.ndarray
            A 3D array containing the molecule counts at each time
            step for all trajectories.
        """
        if trajectories <= 0 or steps <= 0:
            raise ValueError("Trajectories and steps must be positive integers.")

        logger.info(
            f"""Starting simulation: {trajectories}
            trajectories, {steps} steps each."""
        )
        n_molecules = len(self.state.molecules)
        steps = steps + 1
        molecule_counts = np.zeros((trajectories, steps, n_molecules))
        times = np.zeros((trajectories, steps))
        negative_values_detected = False

        for i in trange(trajectories, desc="Simulating trajectories"):
            logger.info(f"Running trajectory {i + 1}/{trajectories}...")
            initial_counts = self.state.extract_molecule_counts()
            if initial_counts is None:
                raise ValueError(
                    f"Failed to extract initial molecule counts for trajectory {i + 1}."
                )
            molecule_counts[i, 0] = initial_counts
            times[i, 0] = 0

            for j in range(1, steps):
                self.state.next_state(self.dt)
                times[i, j] = self.state.time
                counts = self.state.extract_molecule_counts()
                if counts is None:
                    raise ValueError(
                        f"Failed to extract molecule counts at step {j} in trajectory {i + 1}."
                    )
                molecule_counts[i, j] = counts

            # Check for negative counts
            if np.any(counts < 0):
                negative_values_detected = True
                for molecule_num, molecule_name in enumerate(
                    self.state.molecules.keys()
                ):
                    if counts[molecule_num] < 0:
                        logger.warning(
                            f"Negative count detected in trajectory {i + 1}, step {j}, "
                            f"molecule '{molecule_name}': count = {counts[molecule_num]}"
                        )

            # Reset the state for the next trajectory
            self.state.set_init_state()

        if negative_values_detected:
            logger.warning(
                "Negative values were detected in one or more trajectories. "
                "Please reassess the rate parameters to ensure realistic behavior."
                "If decay rates are higher than transcription/translation rates, molecules may deplete too quickly!"
            )
        else:
            logger.info("No negative counts were observed")

        if save:
            if not saven_fname:
                saven_fname = f"molecule_counts_{steps-1}_{trajectories}.npy"
            self.save_runs(molecule_counts, fname=saven_fname, path=save_path)

        self.molecule_counts = molecule_counts
        self.times = times
        return self.molecule_counts

    def plot(
        self,
        example: bool = False,
        scale: str = "linear",
        save_folder: Optional[str] = None,
        show_confidence: bool = True,  # Option to toggle confidence intervals
        dynamic_filename: bool = True,  # Option to use dynamic filenames
    ) -> None:
        """
        Plot the simulation results (state of the system).

        Parameters
        ----------
        example: bool
            If set to True, plots a single random trajectory. Default is False.
        scale: str
            Scale of the y-axis. Default is "linear".
        save_folder: str
            Directory to save the plot. If None, plot is displayed.
        show_confidence: bool
            If True, adds confidence intervals to the plot. Default is True.
        dynamic_filename: bool
            If True, generates a filename based on simulation parameters. Default is True.

        Returns
        -------
        None
        """
        if example:
            rand_num = np.random.randint(0, self.molecule_counts.shape[0])
            for molecule_num, (molecule_name, _) in enumerate(
                self.state.molecules.items()
            ):
                plt.plot(
                    self.molecule_counts[rand_num, :, molecule_num],
                    label=molecule_name,
                )
        else:
            # average over trajectories + confidence intervals
            avg_counts = np.mean(self.molecule_counts, axis=0)
            std_counts = np.std(self.molecule_counts, axis=0)
            mean_times = np.mean(self.times, axis=0)
            for molecule_num, (molecule_name, molecule) in enumerate(
                self.state.molecules.items()
            ):
                molecule_count = avg_counts[:, molecule_num]
                molecule_std = std_counts[:, molecule_num]
                plt.plot(mean_times, molecule_count, label=molecule_name)

                if show_confidence:
                    plt.fill_between(
                        mean_times,
                        molecule_count - molecule_std,
                        molecule_count + molecule_std,
                        alpha=0.2,
                    )
        plt.title(
            f"Simulation of Gene Regulatory Network: {self.molecule_counts.shape[0]} Trajectories with {self.molecule_counts.shape[1] - 1} Steps"
        )
        plt.xlabel("Time (arbitrary units)")
        plt.ylabel("Molecule Count (absolute numbers)")
        plt.yscale(scale)
        plt.legend()

        # checking if output folder path is valid
        if save_folder is not None:
            save_folder = Path(save_folder)

            # creating folder if it doesn't exist
            save_folder.mkdir(parents=True, exist_ok=True)

            # defining save name/path
            if dynamic_filename:
                save_name = (
                    f"simulation_plot_{scale}_{'example' if example else 'all'}.png"
                )
            else:
                save_name = "simulation_plot.png"

            save_path = save_folder.joinpath(save_name)

            # saving plot
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")

        else:
            # just showing plot without saving
            plt.show()
