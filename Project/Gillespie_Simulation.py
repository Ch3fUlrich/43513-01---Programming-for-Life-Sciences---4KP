from __future__ import annotations
from typing import List, Dict, Optional, Union
from argparse import ArgumentParser
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from tqdm import trange
import yaml
from pathlib import Path
import copy


class State_Machine:
    def __init__(
        self,
        init_state_path: Optional[str] = None,
        state: Optional[State] = None,
        dt: int = 1,
    ):
        """
        A state machine object in a cell.

        Parameters:
            init_state_path: Optional[str]
                The initial state path
            state: Optional[State]
                The state of the cell
            dt: int
                The time step
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
        self.count_counts: Optional[np.ndarray] = None

    def save_runs(
        self,
        molecule_counts: np.ndarray,
        fname: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        """
        Save the results of the runs.

        Parameters:
            molecule_counts: np.ndarray
                The number of molecules
            fname: Optional[str]
                The file name
            path: Optional[str]
                The path to save the file
        """
        fname = fname or "molecule_counts"
        path = path or self.path
        fpath = Path(path).parent.joinpath("output", fname)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(fpath, molecule_counts)

    def reset(self) -> None:
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
        saven_fname: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Run the state machine for multiple steps.

        Parameters:
            steps: int
                Number of steps
            trajectories: int
                Number of trajectories
            save: bool
                If True, save the results
            saven_fname: Optional[str]
                The file name to save the results
            save_path: Optional[str]
                The path to save the results
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
            self.state.set_init_state()
        if save:
            if not saven_fname:
                saven_fname = f"molecule_counts_{steps}_{trajectories}.npy"
            self.save_runs(molecule_counts, fname=saven_fname, path=save_path)
        self.molecule_counts = molecule_counts
        self.times = times
        return self.molecule_counts

    def plot(
        self,
        example: bool = False,
        scale: str = "linear",
        save_folder: Optional[str] = None,
    ) -> None:
        """
        Plot the state of the cell.

        Parameters:
            example: bool
                If True, plot an example trajectory
            scale: str
                Scale of the plot
            save_folder: Optional[str]
                Folder to save the plot
        """
        if example:
            rand_num = np.random.randint(0, self.molecule_counts.shape[0])
            for molecule_num, (molecule_name, molecule) in enumerate(
                self.state.molecules.items()
            ):
                plt.plot(
                    self.molecule_counts[rand_num, :, molecule_num],
                    label=molecule_name,
                    self.molecule_counts[rand_num, :, molecule_num],
                    label=molecule_name,
                )
        else:
            avg_counts = np.mean(self.molecule_counts, axis=0)
            std_counts = np.std(self.molecule_counts, axis=0)
            mean_times = np.mean(self.times, axis=0)
            for molecule_num, (molecule_name, molecule) in enumerate(
                self.state.molecules.items()
            ):
                molecule_count = avg_counts[:, molecule_num]
                molecule_std = std_counts[:, molecule_num]
                plt.plot(molecule_count, label=molecule_name)
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

        if save_folder is not None:
            save_folder = Path(save_folder)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_name = "simulation_plot.png"
            save_path = save_folder.joinpath(save_name)
            plt.savefig(save_path)

        else:
            plt.show()


class State:
    def __init__(self, path: str):
        """
        A state object in a cell.

        Parameters:
            path: str
                The path of the state yaml file
        """
        if not isinstance(path, str) and not isinstance(path, Path):
            raise ValueError("Path must be a string or Path object")
        assert Path(path).suffix == ".yaml", "File must be a yaml file"

        self.path: Path = Path(path)
        self.state_dict: Dict[str, Union[str, int, float, List[int]]] = None
        self.time: int = None
        self.molecules: Dict[str, Union[Molecule, Complex]] = None
        self.set_init_state()

    def load_state(self) -> Dict[str, Union[str, int, float, List[int]]]:
        """
        Load the state of the cell from a yaml file.

        Returns:
            Dict[str, Union[str, int, float, List[int]]]:
                The state of the cell as a dictionary
        """
        if self.path:
            with open(self.path, "r") as file:
                state = yaml.safe_load(file)
        else:
            raise ValueError("Path to initial state is not defined")
        return state

    def set_init_state(self) -> None:
        """
        Set the initial state of the cell.
        """
        self.state_dict = self.load_state()
        self.time: int = self.state_dict["time"]
        self.molecules: Dict[str, Union[Molecule, Complex]] = self.create_molecules()

    def save_state(self) -> None:
        """
        Save the state of the cell to a yaml file.
        """
        with open(self.path, "w") as file:
            yaml.dump(self.state_dict, file)

    def create_state_dict(
        self, t: int, save: bool = False
    ) -> Dict[str, Union[str, int, float, List[int]]]:
        """
        Create the state of the cell as a dictionary.

        Parameters:
            t: int
                The time
            save: bool
                If True, save the state to a yaml file

        Returns:
            Dict[str, Union[str, int, float, List[int]]]:
                The state dictionary
        """
        state_dict: Dict[str, Union[str, int, float, List[int]]] = {"time": t}
        for molecule_name, molecule in self.molecules.items():
            molecule_dict = molecule.create_molecule_dict()
            state_dict.update(molecule_dict)
        if save:
            self.save_state()
        return self.state_dict

    def create_molecules(self) -> Dict[str, Union[Molecule, Complex]]:
        """
        Create the molecule objects in the cell.

        Returns:
            Dict[str, Union[Molecule, Complex]]: The molecules in the cell
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

    def next_state(self, dt: int = 1) -> State:
        """
        Update the state of the cell.

        Parameters:
            dt: int
                The time step

        Returns:
            State: The updated state
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

        for molecule_name, m_class in m.items():
            if m_class.count < 0:
                # TIXME: Find the reason behind this bug
                raise ValueError(f"Negative count for {molecule_name}")

        self.molecules = m
        self.create_state_dict(self.state_dict["time"], save=False)
        return self

    def extract_molecule_counts(
        self, as_dict: bool = False
    ) -> Union[List[int], Dict[str, int]]:
        """
        Extract the number of molecules in the cell.

        Parameters:
            as_dict: bool
                If True, return the counts as a dictionary

        Returns:
            Union[List[int], Dict[str, int]]:
                The number of molecules in the cell
        """
        if as_dict:
            counts = {
                molecule_name: molecule.count
                for molecule_name, molecule in self.molecules.items()
            }
        else:
            counts = np.array(
                [molecule.count for _, molecule in self.molecules.items()],
                [molecule.count for _, molecule in self.molecules.items()],
                dtype=int,
            )
        return counts

    def print(self, short: bool = False, full: bool = False) -> None:
        """
        Print the state of the cell.

        Parameters:
            short: bool
                If True, print the short state of the cell
            full: bool
                If True, print the full state of the cell
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


class MoleculeLike:
    def __init__(self, name: str, count: int):
        """
        A molecule like object in a cell

        Parameters:
            name: str
                The name of the molecule
            count: int
                The number of molecules
        """
        self.name = name
        self.count = count

    def create_molecule_dict(self) -> Dict[str, Union[str, int]]:
        """
        Create a dictionary of the molecule object

        Returns:
            Dict[str, Union[str, int]]: The molecule dictionary
        """
        molecule_dict = {}
        for key, value in self.__dict__.items():
            if value:
                molecule_dict[key] = value
        return molecule_dict

    def express(
        self,
        expression_rate: float,
        dt: Optional[int] = None,
        from_count: Optional[int] = None,
    ) -> int:
        """
        Expression of a molecule over time. random choice for every molecule.

        Parameters:
            expression_rate: float
                The rate at which the molecule is expressed
            dt: Optional[int]
                The time
            from_count: Optional[int]
                The initial count of molecules

        Returns:
            int: The number of molecules left after expression
        """
        dt = dt or 1
        count_diff = fast_random_occurrence(expression_rate, from_count)
        return count_diff


class Molecule(MoleculeLike):
    def __init__(
        self,
        name: str,
        count: int,
        translation_rate: Optional[float] = None,
        decay_rate: Optional[float] = None,
        transcription_rate: Optional[float] = None,
        transcription_rate_constant: bool = False,
        k: Optional[float] = None,
        c: Optional[float] = None,
    ):
        """
        A molecule object in a cell.

        Parameters:
            name: str
                The name of the molecule
            count: int
                The number of molecules
            translation_rate: Optional[float]
                The rate at which the molecule is translated
            decay_rate: Optional[float]
                The rate at which the molecule decays
            transcription_rate: Optional[float]
                The rate at which the molecule is created
            transcription_rate_constant: bool
                If True, the transcription_rate value is constant
            k: Optional[float]
                Half-maximal activation constant
            c: Optional[float]
                Hill coefficient for fixing steepness of the activation curve.
                Default value is 1 for linear activation
        """
        super().__init__(name, count)
        self.translation_rate = translation_rate
        self.decay_rate = decay_rate
        self.transcription_rate = transcription_rate
        self.transcription_rate_constant = transcription_rate_constant
        self.k = k
        self.c = c or 1

    def creation_rate(self, q: int, k: float, c: float) -> float:
        """
        General version of the rate equation for the creation of a molecule
        from nothing.

        Parameters:
            q: int
                The number of TF molecules
            k: float
                Half-maximal activation constant
            c: float
                Hill coefficient for fixing steepness of the activation curve.
                Default value is 1 for linear activation
                Hill coefficient for fixing steepness of the activation curve.
                Default value is 1 for linear activation

        Returns:
            float: The new creation rate of the molecule
        """
        c = c or 1
        transcription_rate = self.transcription_rate * (q**c) / (k**c + q**c)
        return transcription_rate

    def decay(self, dt: Optional[int] = None) -> int:
        """
        Decay of a molecule over time

        Parameters:
            dt: Optional[int]
                The time

        Returns:
            int: The number of molecules left after decay
        """
        dt = dt or 1
        return self.express(self.decay_rate, dt, from_count=self.count)

    def transcription(
        self,
        protein: Optional[int] = None,
        dt: Optional[int] = None,
    ) -> int:
        """
        Transcription of a molecule over time

        Parameters:
            protein: Optional[int]
                The number of proteins
            dt: Optional[int]
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

    def translation(self, dt: Optional[int] = None) -> int:
        """
        Translation of a molecule over time

        Parameters:
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
        degradation_rate: Optional[int] = None,
        formation_rate: Optional[int] = None,
    ) -> None:
        """
        A complex object in a cell

        Parameters:
            name (str): The name of the complex
            count (int): The number of complexes
            molecules_per_complex (List[int]):
                The number of molecules needed to form the complex
            degradation_rate (Optional[int]):
                The rate at which the complex degrades
            formation_rate (Optional[int]):
                The rate at which the complex is formed
        """
        super().__init__(name, count)
        self.molecules_per_complex = molecules_per_complex
        self.degradation_rate = degradation_rate
        self.formation_rate = formation_rate

    def degradation(self, dt: Optional[int] = None) -> int:
        """
        Degradation of a molecule over time

        Parameters:
            dt (Optional[int]): The time

        Returns:
            int: The number of molecules left after degradation
        """
        count_diff = self.express(
            self.degradation_rate,
            dt,
            from_count=self.count,
        )
        count_diff = self.express(
            self.degradation_rate,
            dt,
            from_count=self.count,
        )
        return count_diff

    def formation(
        self,
        molecules: List[int],
        dt: Optional[int] = None,
    ) -> List[int]:
        """
        Formation of a complex between multiple molecules over time.
        The complex is formed between multiple molecules 1:1 relationship.

        Parameters:
            molecules (List[int]): The number of molecules
            dt (Optional[int]): The time

        Returns:
            List[int]: The number of molecules left after complex formation
        """
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


def construct_path(path: Optional[str] = None) -> Path:
    """
    Construct the path to the file. If the path is not provided, the current
    working directory is used.
    Construct the path to the file. If the path is not provided, the current
    working directory is used.

    Parameters:
        path (Optional[str]): The path to the file

    Returns:
        Path: The constructed path
    """
    if not path:
        state_folder = "states"
        fname = "init_state.yaml"
        path = Path.cwd().joinpath(state_folder, fname)

    if path.suffix != ".yaml":
        raise ValueError("File must be a yaml file")
    return path


@njit(parallel=True)
def fast_random_occurrence(
    expression_rate: float,
    from_count: int,
) -> np.ndarray:
    """
    Fast random choice for each element in the array.
    This function is 4 times faster than np.random.choice.

    Parameters:
        expression_rate (float): The rate at which the molecule is expressed
        from_count (int): The number of molecules

    Returns:
        np.ndarray: An array of occurrences
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


def get_args_dict() -> dict:
    """
    Parses command-line arguments provided by the user and returns
    them as a dictionary.
    Parses command-line arguments provided by the user and returns
    them as a dictionary.

    This function defines the following arguments:
    - `--initial-state` (`-i`): str, required
        Path to the initial state file (.yaml) that defines the
        simulation's starting conditions.
        Path to the initial state file (.yaml) that defines the
        simulation's starting conditions.
    - `--trajectories` (`-t`): int, optional, default=5
        Number of trajectories for the simulation.
    - `--steps` (`-s`): int, optional, default=100
        Number of steps to run in each trajectory.
    - `--output-folder` (`-o`): str, optional, default="output"
        Path to the folder where the simulation's
        output files (.npy and .png) will be saved.
        Path to the folder where the simulation's
        output files (.npy and .png) will be saved.

    :return: dict - A dictionary containing parsed arguments with argument
        names as keys and user-provided values.
    :return: dict - A dictionary containing parsed arguments with argument
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
        help="""defines path to output folder
            (save .npy and .png simulation plots)""",
    )

    args_dict = vars(parser.parse_args())
    return args_dict


def main() -> None:
    """
    Coordinates the execution of the simulation based on command-line
    arguments.
    Coordinates the execution of the simulation based on command-line
    arguments.

    This function performs the following:
    - Parses arguments using `get_args_dict`.
    - Initializes the initial state using the
        `State` and `State_Machine` classes.
    - Runs the simulation with parameters for
        the number of trajectories and steps.
    - Initializes the initial state using the
        `State` and `State_Machine` classes.
    - Runs the simulation with parameters for
        the number of trajectories and steps.
    - Plots and saves the simulation results if an output folder is specified.

    Workflow:
    1. Parse arguments to retrieve:
        - `input_path`: Path to the initial state file.
        - `output_folder`: Path where outputs will be saved.
        - `save`:
            Boolean indicating whether to save outputs based on `output_folder`
        - `save`:
            Boolean indicating whether to save outputs based on `output_folder`
        - `trajectories`: Number of trajectories to simulate.
        - `steps`: Number of steps in each trajectory.
    2. Run simulation using `State` and `State_Machine` classes.
    3. Plot and save the simulation output.

    :return: None
    """
    args_dict = get_args_dict()
    input_path = args_dict["initial_state"]
    output_folder = args_dict["output_folder"]
    save = output_folder is not None
    trajectories = args_dict["trajectories"]
    steps = args_dict["steps"]

    simulator = State_Machine(init_state_path=input_path)
    simulator.run(
        steps=steps,
        trajectories=trajectories,
        save_path=output_folder,
        save=save,
    )

    simulator.plot(save_folder=output_folder)


if __name__ == "__main__":
    main()
