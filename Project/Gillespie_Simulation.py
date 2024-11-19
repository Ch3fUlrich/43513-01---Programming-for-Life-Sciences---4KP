"""
Simulation Framework for the stochastic dynamics of a simple gene regulatory network 

This module provides a framework for simulating cellular processes using a 
state machine approach. The framework includes:
- Classes for representing molecules, molecular complexes, and simulation states.
- Functions for argument parsing, path construction, and random occurrence simulations.
- A main function that coordinates the execution of simulations.

Modules
-------
- `State_Machine`: Drives the simulation workflow.
- `State`: Manages the current state of the cell, including molecule counts.
- `Molecule` and `Complex`: Represent individual molecules and complexes.
- Utility functions: Includes argument parsing, file path construction, 
  and a fast random occurrence method for molecule events.

Usage
-----
Run the simulation with appropriate command-line arguments:
    python simulation.py -i initial_state.yaml -t 100 -s 100 -o ./output
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting the simulation script...")

try:
    # cli arg parsing
    from argparse import ArgumentParser

    # type hints
    from typing import List, Dict, Optional, Union, Tuple

    # calculation
    import numpy as np

    # fast calculation
    from numba import njit, prange

    # plotting
    import matplotlib.pyplot as plt

    # show time till finished
    from tqdm import trange

    # loading files
    import yaml
    from pathlib import Path

    # copy
    import copy
    
    # Log success
    logger.info("Successfully loaded all required libraries.")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    raise



class State_Machine:
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
    run(steps=100, trajectories=100, save=True, saven_fname=None, save_path=None)
        Runs the simulation for a specified number of steps and trajectories.
    plot(example=False, scale="linear", save_folder=None)
        Generates a plot of the simulation results.
    """
        
    def __init__(self, innit_state_path: str = None, state=None, dt: int = 1) -> None:
        """
        Initializes a state machine for simulation.

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
        if innit_state_path and state is not None:
            raise ValueError("init_state_path and state cannot be provided at the same")
        if innit_state_path:
            self.state = State(innit_state_path)
        else:
            self.state = state
        self.path = self.state.path.parent
        self.dt = dt
        self.times = None
        self.count_counts = None

    def save_runs(
        self, molecule_counts: np.ndarray, fname: str = None, path: str = None
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
            raise
        

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
        saven_fname: str = None,
        save_path: str = None,
    ) -> np.ndarray:
        """
        Runs the state machine for multiple steps and trajectories.

        Parameters
        ----------
        steps : int, optional
            Number of time steps for each trajectory. Default is 100.
        trajectories : int (default is 100)
            Number of independent simulation trajectories to run. Default is 100.
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
            A 3D array containing the molecule counts at each time step for all trajectories.
        """
        logger.info(f"Starting simulation: {trajectories} trajectories, {steps} steps each.")
        n_molecules = len(self.state.molecules)
        steps = steps + 1
        molecule_counts = np.zeros((trajectories, steps, n_molecules))
        molecule_counts[0, 0] = self.state.extract_molecule_counts()
        times = np.zeros((trajectories, steps))
        for i in trange(trajectories):
            logger.info(f"Running trajectory {i + 1}/{trajectories}...")
            for j in range(1, steps):
                self.state.next_state(self.dt)
                times[i, j] = self.state.time
                molecule_counts[i, j] = self.state.extract_molecule_counts()
            # reset the state
            self.state.set_init_state()
        if save:
            if not saven_fname:
                saven_fname = f"molecule_counts_{steps}_{trajectories}.npy"
            self.save_runs(molecule_counts, fname=saven_fname, path=save_path)
        self.molecule_counts = molecule_counts
        self.times = times
        return self.molecule_counts

    def plot(self, example=False, scale="linear", save_folder: str = None) -> None:
        """
        Plots the simulation results (state of the system)

        Parameters
        ---------- 
        example: bool
            If set to True, plots a single random trajectory. Default is False.
        scale: str
            Scale of the y-axis. Default is "linear".
        save_folder: str
            Directory to save the plot. If None, plot is displayed.
        
        Returns
        -------
        None
        """
        if example:
            rand_num = np.random.randint(0, self.molecule_counts.shape[0])
            # Plot the random trajectory
            for molecule_num, (molecule_name, molecule) in enumerate(
                self.state.molecules.items()
            ):
                plt.plot(
                    self.molecule_counts[rand_num, :, molecule_num], label=molecule_name
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
                plt.plot(molecule_count, label=molecule_name)
                # confidence intervals
                plt.fill_between(
                    mean_times,
                    molecule_count - molecule_std,
                    molecule_count + molecule_std,
                    alpha=0.2,
                )

        plt.yscale(scale)
        plt.xlabel("Time")
        plt.ylabel(f"Molecule count ({scale})")
        plt.legend()

        # checking if output folder path is valid
        if save_folder is not None:
            save_folder = Path(save_folder)

            # creating folder if it doesn't exist
            save_folder.mkdir(parents=True, exist_ok=True)

            # defining save name/path
            save_name = f"simulation_plot.png"
            save_path = save_folder.joinpath(save_name)

            # saving plot
            plt.savefig(save_path)

        else:

            # just showing plot wo saving
            plt.show()


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
        Initializes the state from a YAML file.

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

        self.path = Path(path)
        self.state_dict: Dict = None
        self.time: int = None
        self.molecules: Dict = None
        self.set_init_state()
        logger.info("State initialization completed successfully.")

    def load_state(self) -> dict:
        """
        Loads the state of the cell from the specified YAML file and 
        returns it as a dictionary.

        Returns
        -------
        dict: 
            The state of the cell as a dictionary
        
        Raises
        ------
        ValueError
            If the provided `path` is not a string or a `Path` object, or if it 
            does not point to a valid `.yaml` file.
        """
        if self.path:
            with open(self.path, "r") as file:
                state = yaml.safe_load(file)
        else:
            raise ValueError("Path to initial state is not defined")
        return state

    def set_init_state(self) -> None:
        """
        Initializes the state by loading the YAML file and setting up 
        the initial simulation state.

        This method loads the state dictionary from the YAML file using 
        `load_state()` and initializes the simulation time and molecules 
        based on the loaded data.

        Attributes
        ----------
        state_dict : dict
            The dictionary containing the full state information loaded from the YAML file.
        time : int
            The current simulation time, extracted from the loaded state dictionary.
        molecules : dict
            A dictionary of molecules in the state, created using the `create_molecules()` method. 
            This contains molecule names as keys and their properties as values.

        Returns
        -------
        None
        """
        self.state_dict = self.load_state()
        self.time: int = self.state_dict["time"]
        self.molecules: Dict = self.create_molecules()

    def save_state(self) -> None:
        """
        Saves the current state of the cell to a YAML file.

        This method writes the `state_dict` attribute, which contains the current 
        state of the cell, to the specified YAML file defined in `self.path`.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        with open(self.path, "w") as file:
            yaml.dump(self.state_dict, file)

    def create_state_dict(self, t, save: bool = False) -> dict:
        """
        Creates and updates the state of the cell as a dictionary.
        Method generates the current state of the cell (including the 
        simulation time) and the states of all molecules. Optionally, if save set to True,
        state is saved to a YAML file.

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
        state_dict = {"time": t}
        for molecule_name, molecule in self.molecules.items():
            molecule_dict = molecule.create_molecule_dict()
            state_dict.update(molecule_dict)
        if save:
            self.save_state()
        return self.state_dict

    def create_molecules(self) -> dict:
        """
        Creates molecule objects based on the initial state.

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

    def next_state(self, dt: int = 1):
        """
        This method updates the simulation time by one time step, calculates changes in molecule 
        counts based on transcription, decay, 
        translation, and complex formation and updates the state dictionary with 
        the new molecule counts. The updated state object is returned.

        Parameters
        ----------
        dt : int, optional
            The time step for the simulation. Default is 1.

        Returns
        -------
        state:
            The updated state object after applying the changes for the next time step.
        """
        # 1. Update the time
        self.time = self.time + dt

        # 2. Calculate changes in time for each molecule
        m = copy.deepcopy(self.molecules)

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
        formed_complex, used_molecules = m["complex"].formation(
            [m["miRNA"].count, m["mRNA"].count], dt
        )
        complex_degraded = m["complex"].degradation(dt)

        m["TF_mRNA"].count = m["TF_mRNA"].count + TF_mRNA_trancribed - TF_mRNA_decayed
        m["TF_protein"].count = (
            m["TF_protein"].count + TF_mRNA_translated - TF_protein_decayed
        )
        m["miRNA"].count = (
            m["miRNA"].count
            + miRNA_trancribed
            - miRNA_decayed
            - used_molecules[0]
            + complex_degraded
        )
        m["mRNA"].count = (
            m["mRNA"].count + mRNA_trancribed - mRNA_decayed - used_molecules[1]
        )
        m["protein"].count = m["protein"].count + mRNA_translated - protein_decayed
        m["complex"].count = m["complex"].count + formed_complex - complex_degraded

        # Debugging Output
        # if self.time == 1000:
        #    self.print(short=True)
        #    print("1000")

        self.molecules = m
        self.create_state_dict(self.state_dict["time"], save=False)
        # self.print(short=True)
        return self

    def extract_molecule_counts(self, as_dict: bool = False) -> Union[Dict[str, int], np.ndarray]:
        """
        Extracts molecule counts as a dictionary or NumPy array.

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
            If `as_dict=True`: A dictionary with molecule names as keys and counts as values.
            If `as_dict=False`: A NumPy array of molecule counts.
        """
        if as_dict:
            counts = {
                molecule_name: molecule.count
                for molecule_name, molecule in self.molecules.items()
            }
        else:
            counts = np.array(
                [molecule.count for molecule_name, molecule in self.molecules.items()],
                dtype=int,
            )
        return counts

    def print(self, short: bool = False, full: bool = False) -> None:
        """
        Prints the current state of the cell in three possible formats: short, full, or default.

        Parameters
        ----------
        short: bool
            If True, prints only the molecule counts. Default is False.
        full: bool
            If True, print the full details of state of the cell. Default is False.
        
        Returns
        -------
        None
        """
        current_time = self.time
        print("------------------------")
        print(f"State(t={current_time})")
        if full:
            for name, key_values in self.state_dict.items():
                if isinstance(key_values, dict):
                    print(f"- {name}:")
                    for attribute, value in key_values.items():
                        print(f"   - {attribute}: {value}")
                else:
                    print(f"- {name}: {key_values}")
        elif short:
            print(self.extract_molecule_counts())
        else:

            for molecule_name, molecule in self.molecules.items():
                print(f"-> {molecule_name}: {molecule.count}")
                # plotting the full molecule objects is too verbose
                # print(f"-> {molecule.name}: {molecule.__dict__}")


class MoleculeLike:
    """
    A base class representing a molecule-like object

    Attributes
    ----------
    name : str
        The name of the molecule
    count : int
        The initial count of the molecule

    Methods
    -------
    create_molecule_dict()
        Creates a dictionary of the molecule
    express(expression_rate, dt=None, from_count=None)
        Simulates the expression or decay of a molecule over time
    """

    def __init__(self, name: str, count: int) -> None :
        """
        Initialize a molecule-like object.
    
        Parameters
        ----------
        name: str
            name of molecule
        count: int
            The initial number of molecules
        """
        self.name = name
        self.count = count

    def create_molecule_dict(self) -> dict:
        """
        Create a dictionary of the molecule object

        Returns
        -------
        dict:
            A dictionary containing the molecule's attributes.
        """
        molecule_dict = {}
        for key, value in self.__dict__.items():
            if value:
                molecule_dict[key] = value
        return molecule_dict

    def express(
        self, expression_rate: int, dt: int = None, from_count: int = None
    ) -> int:
        """
        Simulates the expression or decay of a molecule over time. Random choice for every molecule.
        
        Parameters
        ----------
        expression_rate: int
            The rate at which the molecule is expressed/decayed.
        dt: int
            the time step. Default is 1 (or dt)
        from_count: int, optional
            The initial count of molecules involved in the process.

        Returns
        -------
        int: 
            The change in the number of molecules left after expression/decay
        """
        # Default time step is 1
        dt = dt or 1
        # Randomly choose if something happens for each molecule
        count_diff = fast_random_occurrence(expression_rate, from_count)
        return count_diff


class Molecule(MoleculeLike):
    """
    Represents a molecule in a cell simulation.

    Extends `MoleculeLike` to include additional attributes for transcription, 
    translation, and decay processes.

    Attributes
    ----------
    translation_rate : float
        The rate at which the molecule is translated.
    decay_rate : float
        The rate at which the molecule decays.
    transcription_rate : float
        The rate at which the molecule is transcribed.
    transcription_rate_constant : bool
        Whether the transcription rate is constant.
    k : float
        The half-maximal activation constant.
    c : float
        The Hill coefficient controlling the steepness of the activation curve.

    Methods
    -------
    creation_rate(q, k, c)
        Calculates the molecule's creation rate.
    decay(dt=None)
        Simulates molecule decay over time.
    transcription(protein=None, dt=None)
        Simulates transcription of the molecule over time.
    translation(dt=None)
        Simulates translation of the molecule over time.
    """

    def __init__(
        self,
        name: str,
        count: int,
        translation_rate: float = None,
        decay_rate: float = None,
        transcription_rate: float = None,
        transcription_rate_constant: bool = False,
        k: float = None,
        c: float = None,
    ) -> None:
        """
        Initializes a molecule object with its attributes.

        Parameters
        ----------
        name: str
            name of the molecule
        count: int
            the initial number of molecules
        translation_rate: float, optional
            The rate at which the molecule is translated
        decay_rate: float, optional
            The rate at which the molecule decays
        transcription_rate: float, optional
            The rate at which the molecule is created
        transcription_rate_constant: bool, optional
            If True, the transcription_rate value is constant
        k: float
            Half-maximal activation constant
        c: float
            Hill coefficient for fixing steepness of the activation curve. Default value is 1 for linear activation
        """
        super().__init__(name, count)
        self.translation_rate = translation_rate
        self.decay_rate = decay_rate
        self.transcription_rate = transcription_rate
        self.transcription_rate_constant = transcription_rate_constant
        self.k = k
        # Hill coefficient set to 1 for linear activation if not provided
        self.c = c or 1

    def creation_rate(self, q: int, k: float, c: float) -> float:
        """
        Calculate the rate of molecule creation using a generalized Hill function.

        This method computes the creation rate of a molecule based on the number of molecules (`q`), the half-maximal activation constant (`k`), and the steepness 
        of the activation curve controlled by the Hill coefficient (`c`). The equation is given by:

        new_transcription_rate(Q) = transcription_rate * (q^c / (q^c + k^c))

        Parameters
        ----------
        q: int
            The number of TF molecules
        k: float
            Half-maximal activation constant
        c: float
            The Hill coefficient, controlling the steepness of the activation curve. 
            Default is 1, which corresponds to linear activation.
        Returns
        -------
        float: 
            The new creation rate of the molecule
        """
        c = c or 1
        transcription_rate = self.transcription_rate * (q**c) / (k**c + q**c)
        return transcription_rate

    def decay(self, dt: int = None) -> int:
        """
        Decay of a molecule over time
        
        Parameters:
        dt: int
            The time step

        Returns
        -------
        int: 
            The number of molecules left after decay
        """
        # Default time step is 1
        dt = dt or 1
        return self.express(self.decay_rate, dt, from_count=self.count)

    def transcription(self, protein: int = None, dt: int = None) -> int:
        """
        Simulates molecule transcription over time.

        Parameters
        ----------
        protein: int, optional
            The number of protein molecules influencing transcription. If the transcription rate is 
            non-constant, this value must be provided.
        dt: int
            The time step over which transcription is simulated. Defaults to 1 if not provided.

        Returns
        -------
        int: 
            The number of molecules transcribed.

        Raises
        ------
        ValueError
            If `protein` is not provided and the transcription rate is not constant.
        """
        if protein is None and not self.transcription_rate_constant:
            raise ValueError(
                "Protein must be provided for non-constant transcription rate"
            )
        if self.transcription_rate_constant:
            transcription_rate = self.transcription_rate
            from_count = 1
        else:
            transcription_rate = self.creation_rate(protein, self.k, self.c)
            from_count = protein
        return self.express(transcription_rate, dt, from_count=from_count)

    def translation(self, dt: int = None) -> int:
        """
        Simulates molecule translation over time.

        Parameters
        ----------
        dt: int, optional
            The time step over which translation is simulated. If not provided, 
            it defaults to 1.

        Returns
        -------
        int: 
            The number of molecules translated.
        """
        return self.express(self.translation_rate, dt, from_count=self.count)


class Complex(MoleculeLike):
    """
    Represents complex-formation in the simulation.

    Extends `MoleculeLike` to include additional attributes for formation 
    and degradation processes.

    Attributes
    ----------
    molecules_per_complex : List[int]
        The number of molecules required to form the complex.
    degradation_rate : int
        The rate at which the complex degrades.
    formation_rate : int
        The rate at which the complex forms.

    Methods
    -------
    degradation(dt=None)
        Simulates complex degradation over time.
    formation(molecules, dt=None)
        Simulates the formation of a complex over time.
    """
    def __init__(
        self,
        name: str,
        count: int,
        molecules_per_complex: List[int],
        degradation_rate: int = None,
        formation_rate: int = None,
    ) -> None:
        """
       Initializes a complex object with its attributes.

        Parameters
        ----------
        name: str
            name of molecule-complex
        count: int
            The intial number of complexes
        molecules_per_complex: List[int]
            The number of molecules needed to form the complex
        degradation_rate: int
            The rate at which the complex degrades
        formation_rate: int
            The rate at which the complex is formed
        """
        super().__init__(name, count)
        self.molecules_per_complex = molecules_per_complex
        self.degradation_rate = degradation_rate
        self.formation_rate = formation_rate

    def degradation(self, dt: int = None) -> int:
        """
        Simulates degradation of a molecule over time
        
        Parameters
        ----------
        dt: int, optional
            The time step for the degradation simulation. Default is 1.

        Returns
        -------
        int: 
            The number of molecules degraded.
            #### clarify whether the return value represents the number of molecules degraded or remaining
        """
        count_diff = self.express(self.degradation_rate, dt, from_count=self.count)
        return count_diff

    def formation(self, molecules: List[int], dt: int = None) -> List[int]:
        """
        Simulates the formation of a complex over time.
        The complex is formed between multiple molecules in a 1:1 relationship.
        
        Parameters
        ----------
        molecules: List[int]
            The counts of molecules available for complex formation.
        dt: int
            The time step for the formation simulation. Defaults to 1.

        Returns
        -------
        List[int]:
            The number of complexes formed and the change in molecule counts.
        """
        # Default time step is 1
        dt = dt or 1
        # Default number of molecules is 1 for each molecule
        num_possible_new_complexes = min(
            np.array(molecules) // np.array(self.molecules_per_complex)
        )
        formed_complexes = self.express(
            self.formation_rate, dt, from_count=num_possible_new_complexes
        )
        other_count_change = np.array(self.molecules_per_complex) * formed_complexes
        return formed_complexes, other_count_change


def construct_path(path: str = None, fname: str = None) -> str:
    """
    Constructs a file path for suimulation states.
    If the path is not provided, the current working directory is used.

    Parameters
    ----------
    path : str, optional
        The directory path to the file. Defaults to the current working directory.
    fname : str, optional
        The name of the file. Defaults to "init_state.yaml".

    Returns
    -------
    str
        The constructed file path.

    Raises
    ------
    ValueError
        If the constructed file path does not have a ".yaml" extension.
    """
    path = path or Path.cwd()
    # TODO: check if it's better to keep folder+path construction here or if take the whole path from cli
    fname = fname or "init_state.yaml"
    fpath = Path(path).joinpath("states", fname)
    if fpath.suffix != ".yaml":
        raise ValueError("File must be a yaml file")
    return fpath


@njit(parallel=True)
def fast_random_occurrence(expression_rate: float, from_count: int) -> np.ndarray:
    """
    Simulates random occurrences for molecules at a given expression rate.
    This function is 4 times faster than np.random.choice.

    Parameters
    ----------
    expression_rate: float
        The probability rate at which a molecule is expressed, typically between 0 and 1.
    from_count: int
        The number of molecules

    Returns
    -------
    int
        The total number of molecules that are expressed.
    """
    # Precompute the cumulative probability
    cumulative_prob = 1 - expression_rate
    occurences = np.zeros(from_count, dtype=np.int32)

    # Loop through each element and assign based on random value
    for i in prange(from_count):
        rand_val = np.random.rand()  # Generate a random float in [0, 1)
        if rand_val > cumulative_prob:
            occurences[i] = 1

    num_occurences = np.sum(occurences)
    return num_occurences


def get_args_dict() -> dict:
    """
    Parses command-line arguments provided by the user and returns them as a dictionary.

    This function defines the following arguments:
    - `--initial-state` (`-i`): str, required
        Path to the initial state file (.yaml) that defines the simulation's starting conditions.
    - `--trajectories` (`-t`): int, optional, default=5
        Number of trajectories for the simulation.
    - `--steps` (`-s`): int, optional, default=100
        Number of steps to run in each trajectory.
    - `--output-folder` (`-o`): str, optional, default="output"
        Path to the folder where the simulation's output files (.npy and .png) will be saved.

    Returns
    ------
    dict:
        A dictionary containing parsed arguments with argument names as keys and user-provided values.
    """
    # defining program description
    description = "run gillespie simulation"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # initial state param
    parser.add_argument(
        "-i",
        "--initial-state",
        dest="initial_state",
        type=str,
        required=False,
        help="defines path to initial state file (.yaml)",
    )

    # trajectories param
    parser.add_argument(
        "-t",
        "--trajectories",
        dest="trajectories",
        type=int,
        required=False,
        default=5,
        help="defines number of trajectories",
    )

    # steps param
    parser.add_argument(
        "-s",
        "--steps",
        dest="steps",
        type=int,
        required=False,
        default=100,
        help="defines number of steps",
    )

    # output folder param
    parser.add_argument(
        "-o",
        "--output-folder",
        dest="output_folder",
        type=str,
        required=False,
        default="output",
        help="defines path to output folder (save .npy and .png simulation plots)",
    )

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


def main() -> None:
    """
    Coordinates the execution of the simulation based on user input in command-line.

    This function performs the following:
    - Parses arguments using `get_args_dict`.
    - Initializes the initial state using the `State` and `State_Machine` classes.
    - Runs the simulation with Args for the number of trajectories and steps.
    - Plots and saves the simulation results if an output folder is specified.

    Workflow
    --------
    1. Parse arguments to retrieve:
        - `input_path`: Path to the initial state file.
        - `output_folder`: Path where outputs will be saved.
        - `save`: Boolean indicating whether to save outputs based on `output_folder`.
        - `trajectories`: Number of trajectories to simulate.
        - `steps`: Number of steps in each trajectory.
    2. Run simulation using `State` and `State_Machine` classes.
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

    # running simulation
    init_state_path = construct_path(fname=input_path)
    logger.info(f"Resolved initial state path: {init_state_path}")

    start_state = State(init_state_path)
    logger.info("Initialized state machine.")

    simulator = State_Machine(state=start_state)
    results = simulator.run(
        steps=steps, trajectories=trajectories, save_path=output_folder, save=save
    )
    logger.info("Simulation completed successfully.")

    # plotting/saving simulation
    simulator.plot(save_folder=output_folder)
    logger.info("Plotting completed.")


if __name__ == "__main__":
    main()
