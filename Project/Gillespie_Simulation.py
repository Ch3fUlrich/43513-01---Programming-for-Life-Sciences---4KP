print("importing required libraries...")  # noqa
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


class State_Machine:
    def __init__(self, innit_state_path: str = None, state=None, dt: int = 1):
        """
        A state machine object in a cell.

        Args:
            innit_state_path: str 
                The path to the initial state YAML file
            state: State
                The state of the cell
            dt: int
                The time step for the simulation
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
    ):
        """
        Save the results of the runs.

        Args:
            molecule_counts: np.ndarray
                The number of molecules
            fname: str
                The file name
            path: str
                The path to save the file
        """
        # Default file name
        fname = fname or "molecule_counts"
        path = path or self.path
        fpath = Path(path).parent.joinpath("output", fname)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(fpath, molecule_counts)

    def reset(self):
        """
        Reset the state of the cell.
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
    ):
        """
        Run the state machine for multiple steps.

        Args:
            steps: int (default is 100)
                Number of steps
            trajectories: int (default is 100)
                Number of trajectories
            save: bool
                Whether to save the results. Default is True (=save results).
            saven_fname: str
                The file name to save the results
            save_path: str
                The path to save the results
        Returns: numpy.ndarray
            The molecule counts from the simulation
        """
        n_molecules = len(self.state.molecules)
        steps = steps + 1
        molecule_counts = np.zeros((trajectories, steps, n_molecules))
        molecule_counts[0, 0] = self.state.extract_molecule_counts()
        times = np.zeros((trajectories, steps))
        for i in trange(trajectories):
            print(f"Trajectory {i+1}/{trajectories}")
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
        Plots the state of the cell.

        Args: 
            example: bool
                If set to True, plots a single random trajectory. Default is False.
            scale: str
                Scale of the y-axis. Default is "linear".
            save_folder: str
                Directory to save the plot. If None, displays the plot.
        
        Returns:
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
    def __init__(self, path: str):
        """
        A state object in a cell.

        Args:
            path: str
                The path of the state yaml file
       
       Attributes:
        path: str
            The path to the state YAML file.
        state_dict: dict
            Dictionary containing the state information
        time: int
            time of current simulation
        molecules: dict
            Dictionary of molecules in the state.
        """
        if not isinstance(path, str) and not isinstance(path, Path):
            raise ValueError("Path must be a string or Path object")
        assert Path(path).suffix == ".yaml", "File must be a yaml file"

        self.path = Path(path)
        self.state_dict: Dict = None
        self.time: int = None
        self.molecules: Dict = None
        self.set_init_state()

    def load_state(self) -> dict:
        """
        Load the state of the cell from a yaml file.

        Returns:
            dict: The state of the cell as a dictionary
        """
        if self.path:
            with open(self.path, "r") as file:
                state = yaml.safe_load(file)
        else:
            raise ValueError("Path to initial state is not defined")
        return state

    def set_init_state(self):
        """
        Sets the initial state of the cell.

        Returns:
            None
        """
        self.state_dict = self.load_state()
        self.time: int = self.state_dict["time"]
        self.molecules: Dict = self.create_molecules()

    def save_state(self):
        """
        Saves the state of the cell to a yaml file.

        Returns:
            None
        """
        with open(self.path, "w") as file:
            yaml.dump(self.state_dict, file)

    def create_state_dict(self, t, save: bool = False) -> dict:
        """
        Creates the state of the cell as a dictionary.

        Args:
            t: int
                Time of current simulation
            save: bool
                If True, save the state to a YAML file
        Returns:
            dict: The updated state dictionary.
        """
        state_dict = {"time": t}
        for molecule_name, molecule in self.molecules.items():
            molecule_dict = molecule.create_molecule_dict()
            state_dict.update(molecule_dict)
        if save:
            self.save_state()
        return self.state_dict

    def create_molecules(self):
        """
        Create the molecule objects in the cell.

        Returns:
            dict: A dictionary of molecule objects.
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
        Updates the state of the cell for the next time step 

        Args:
            dt: int
                The time step for next. Default is 1.
        Returns:
            State: The updated state object.
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

    def extract_molecule_counts(self, as_dict: bool = False) -> List[int]:
        """
        Extract the number of molecules in the cell.

        Args: 
            as_dict: bool
                if True, returns dictionary with molecule counts. Default is False.

        Returns:
            List[int]: The number of molecules in the cell
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

    def print(self, short: bool = False, full: bool = False):
        """
        Print the state of the cell.

        Args:
            short: bool
                If True, prints only the molecule counts. Default is False.
            full: bool
                If True, print the full details of state of the cell. Default is False.
        
        Returns:
            None.
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
    def __init__(self, name: str, count: int):
        """
        A molecule like object in a cell
        
        Args:
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

        Returns: dict
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
        Args:
            expression_rate: int
                The rate at which the molecule is expressed/decayed.
            dt: int
                the time step. Default is 1 (or dt)
            from_nothing: bool
                If True, the expression rate is from nothing, so count is 1

        Returns:
            int: The change in the number of molecules left after expression/decay
        """
        # Default time step is 1
        dt = dt or 1
        # Randomly choose if something happens for each molecule
        count_diff = fast_random_occurrence(expression_rate, from_count)
        return count_diff


class Molecule(MoleculeLike):
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
    ):
        """
        A molecule object in a cell.

        Args:
            name: str
                name of the molecule
            count: int
                the initial number of molecules
            translation_rate: float
                The rate at which the molecule is translated
            decay_rate: float
                The rate at which the molecule decays
            transcription_rate: float
                The rate at which the molecule is created
            transcription_rate_constant: bool
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
        General version of the rate equation for the creation of a molecule from nothing.
        $\new_transcription_rate(Q) = \transcription_rate \frac{q}{q+K}$

        Args:
            q: int
                The number of TF molecules
            k: float
                Half-maximal activation constant
            c: float
                Hill coefficient for fixing steepness of the activation curve. Default value is 1 for linear activation

        Returns:
            float: The new creation rate of the molecule
        """
        # Hill coefficient set to 1 for linear activation if not provided
        c = c or 1
        transcription_rate = self.transcription_rate * (q**c) / (k**c + q**c)
        return transcription_rate

    def decay(self, dt: int = None) -> int:
        """
        Decay of a molecule over time
        
        Args:
            dt: int
                The time step

        Returns:
            int: The number of molecules left after decay
        """
        # Default time step is 1
        dt = dt or 1
        return self.express(self.decay_rate, dt, from_count=self.count)

    def transcription(self, protein: int = None, dt: int = None) -> int:
        """
        Transcription of a molecule over time

        Args:
            protein: int
                The number of proteins
            from_nothing: bool
                If True, the transcription rate is from nothing, so count is 1
            dt: int
                The time

        Returns:
            int: The number of molecules left after transcription
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
        Translation of a molecule over time

        Args:
            dt: int
                The time

        Returns:
            int: The number of molecules left after translation
        """
        return self.express(self.translation_rate, dt, from_count=self.count)


class Complex(MoleculeLike):
    def __init__(
        self,
        name: str,
        count: int,
        molecules_per_complex: List[int],
        degradation_rate: int = None,
        formation_rate: int = None,
    ):
        """
        A complex object in a cell

        Args:
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

        Returns:
            int: The number of molecules left after degradation
        """
        super().__init__(name, count)
        self.molecules_per_complex = molecules_per_complex
        self.degradation_rate = degradation_rate
        self.formation_rate = formation_rate

    def degradation(self, dt: int = None) -> int:
        """
        Degradation of a molecule over time
        Args:
            dt: int
                The time

        Returns:
            int: The number of molecules left after degradation
        """
        count_diff = self.express(self.degradation_rate, dt, from_count=self.count)
        return count_diff

    def formation(self, molecules: List[int], dt: int = None) -> List[int]:
        """
        Formation of a complex between multiple molecules over time.
        The complex is formed between multiple molecules 1:1 relationship.
        
        Args:
            molecules: List[int]
                The number of molecules
            num_molecules: List[int]
                The number of molecules needed to form the complex
            complex_rate: int
                The rate at which the complex is formed
            dt: int
                The time

        Returns:
            List[int]: The number of molecules left after complex formation
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
    Construct the path to the file. If the path is not provided, the current working directory is used.

    Args:

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
    Fast random choice for each element in the array.
    This function is 4 times faster than np.random.choice.

    Args:
        expression_rate: float
            The rate at which the molecule is expressed
        from_count: int
            The number of molecules
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

    :return: dict - A dictionary containing parsed arguments with argument names as keys and user-provided values.
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


def main():
    """
    Coordinates the execution of the simulation based on command-line arguments.

    This function performs the following:
    - Parses arguments using `get_args_dict`.
    - Initializes the initial state using the `State` and `State_Machine` classes.
    - Runs the simulation with Args for the number of trajectories and steps.
    - Plots and saves the simulation results if an output folder is specified.

    Workflow:
    1. Parse arguments to retrieve:
        - `input_path`: Path to the initial state file.
        - `output_folder`: Path where outputs will be saved.
        - `save`: Boolean indicating whether to save outputs based on `output_folder`.
        - `trajectories`: Number of trajectories to simulate.
        - `steps`: Number of steps in each trajectory.
    2. Run simulation using `State` and `State_Machine` classes.
    3. Plot and save the simulation output.

    :return: None
    """
    # parsing args
    args_dict = get_args_dict()
    input_path = args_dict["initial_state"]
    output_folder = args_dict["output_folder"]
    save = output_folder is not None
    trajectories = args_dict["trajectories"]
    steps = args_dict["steps"]

    # running simulation
    init_state_path = construct_path(fname=input_path)
    start_state = State(init_state_path)
    simulator = State_Machine(state=start_state)
    results = simulator.run(
        steps=steps, trajectories=trajectories, save_path=output_folder, save=save
    )

    # plotting/saving simulation
    simulator.plot(save_folder=output_folder)


if __name__ == "__main__":
    main()
