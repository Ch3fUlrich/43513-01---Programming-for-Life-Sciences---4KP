# type hints
from typing import List, Dict

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
    def __init__(self, innit_state_path:str=None, state=None, dt:int=1):
        """
        A state machine object in a cell.

        Parameters:
            state: State
                The state of the cell
            dt: int
                The time step
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

    def save_runs(self, molecule_counts:np.ndarray, fname:str=None, path:str=None):
        """
        Save the results of the runs.

        Parameters:
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
        fpath = Path(path).joinpath(fname)
        np.save(fpath, molecule_counts)

    def reset(self):
        """
        Reset the state of the cell.
        """
        self.state = State(self.state.path)
        self.times = None
        self.molecule_counts = None

    def run(self, steps:int=100, trajectories:int=100, save=True, 
            saven_fname:str=None, save_path:str=None):
        """
        Run the state machine for multiple steps.

        Parameters:
            steps: int
                Number of steps
            trajectories: int
                Number of trajectories
            save: bool
                If True, save the results
            saven_fname: str
                The file name to save the results
            save_path: str
                The path to save the results
        """
        n_molecules = len(self.state.molecules)
        molecule_counts = np.zeros((trajectories, steps+1, n_molecules))
        times = np.zeros((trajectories, steps+1))
        for i in range(trajectories):
            print(f"Trajectory {i+1}/{trajectories}")
            for j in trange(steps):
                self.state.next_state(self.dt)
                times[i, j] = self.state.time
                molecule_counts[i, j] = self.state.extract_molecule_counts()
        if save:
            if not saven_fname:
                saven_fname = f"molecule_counts_{steps}_{trajectories}.npy"
            self.save_runs(molecule_counts, fname=saven_fname, path=save_path)
        self.molecule_counts = molecule_counts
        self.times = times
        return self.molecule_counts

    def plot(self):
        """
        Plot the state of the cell.
        """

        #average over trajectories + confidence intervals
        avg_counts = np.mean(self.molecule_counts, axis=0)
        std_counts = np.std(self.molecule_counts, axis=0)
        mean_times = np.mean(self.times, axis=0)
        for molecule_num, (molecule_name, molecule) in enumerate(self.state.molecules.items()):
            molecule_count = avg_counts[:, molecule_num]
            molecule_std = std_counts[:, molecule_num]
            plt.plot(molecule_count, label=molecule_name)
            # confidence intervals
            plt.fill_between(mean_times, molecule_count - molecule_std, molecule_count + molecule_std, alpha=0.2)

        plt.xlabel("Time")
        plt.ylabel("Molecule count")
        plt.legend()
        plt.show()

class State:
    def __init__(self, path:str):
        """
        A state object in a cell.

        Parameters:
            path: str
                The path of the state yaml file
            molecules: dict

        """
        if not isinstance(path, str) and not isinstance(path, Path):
            raise ValueError("Path must be a string or Path object")
        assert Path(path).suffix == '.yaml', "File must be a yaml file"

        self.path = Path(path)
        self.state_dict = self.load_state()
        self.time = self.state_dict["time"]
        self.molecules:Dict = self.create_molecules()

    def load_state(self) -> dict:
        """
        Load the state of the cell from a yaml file.

        Returns:
            dict: The state of the cell as a dictionary
        """
        if self.path:
            with open(self.path, 'r') as file:
                state = yaml.safe_load(file)
        else:
            raise ValueError("Path not provided")
        return state

    def save_state(self):
        """
        Save the state of the cell to a yaml file.
        """
        with open(self.path, 'w') as file:
            yaml.dump(self.state_dict, file)

    def create_state_dict(self, t, save:bool=False) -> dict:
        """
        Create the state of the cell as a dictionary.

        Parameters:
            save: bool
                If True, save the state to a yaml file
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


    def next_state(self, dt:int=1):
        """
        Update the state of the cell.

        Parameters:
            dt: int
                The time step
        """
        # 1. Update the time
        self.time += dt

        # 2. Calculate changes in time for each molecule
        #FIXME: find error in calculation of molecule changes
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
        formed_complex, used_molecules = m["complex"].formation([m["miRNA"].count, m["mRNA"].count], dt)
        complex_degraded = m["complex"].degradation(dt)
        
        # 3. Update the number of molecules
        m["TF_mRNA"] + TF_mRNA_trancribed - TF_mRNA_decayed
        m["TF_protein"] + TF_mRNA_translated - TF_protein_decayed
        m["miRNA"] + miRNA_trancribed - miRNA_decayed - used_molecules[0] + complex_degraded
        m["mRNA"] + mRNA_trancribed - mRNA_decayed - used_molecules[1]
        m["protein"] + mRNA_translated - protein_decayed
        m["complex"] + formed_complex - complex_degraded

        if self.time == 1000:
            self.print(short=True)
            print("1000")
            
        self.molecules = m
        self.create_state_dict(self.state_dict["time"], save=False)
        #self.print(short=True)
        return self
    
    def extract_molecule_counts(self, as_dict:bool=False) -> List[int]:
        """
        Extract the number of molecules in the cell.

        Returns:
            List[int]: The number of molecules in the cell
        """
        if as_dict:
            counts = {molecule_name: molecule.count for molecule_name, molecule in self.molecules.items()}
        else:
            counts = np.array([molecule.count for molecule_name, molecule in self.molecules.items()], dtype=int)
        return counts
    
    def print(self, short:bool=False, full:bool=False):
        """
        Print the state of the cell.

        Parameters:
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
                # plotting the full molecule objects is too verbose
                #print(f"-> {molecule.name}: {molecule.__dict__}")

class MoleculeLike:
    def __init__(self, name:str, count:int):
        """
        A molecule like object in a cell
        Parameters:
            count: int
                The number of molecules
        """ 
        self.name = name
        self.count = count

    def __add__(self, molecule_change:int) -> int:
        """
        Add the number of molecules. Overloading the + operator.

        Returns:
            int: The number of molecules left after addition
        """
        self.count += molecule_change
        return self.count
    
    def __sub__(self, molecule_change:int) -> int:
        """
        Subtract the number of molecules. Overloading the - operator.

        Returns:
            int: The number of molecules left after subtraction
        """
        self.count -= molecule_change
        return self.count
    
    def create_molecule_dict(self) -> dict:
        """
        Create a dictionary of the molecule object
        """
        molecule_dict = {}
        for key, value in self.__dict__.items():
            if value:
                molecule_dict[key] = value
        return molecule_dict

    def express(self, expression_rate:int, dt:int=None, from_count:int=None) -> int:
        """
        Expression of a molecule over time. random choice for every molecule.
        Parameters:
            expression_rate: int
                The rate at which the molecule is expressed
            from_nothing: bool
                If True, the expression rate is from nothing, so count is 1
            dt: int
                The time
        
        Returns:
            int: The number of molecules left after expression
        """
        # Default time step is 1
        dt = dt or 1
        # Randomly choose if something happens for each molecule
        count_diff = fast_random_occurrence(expression_rate, from_count)
        return count_diff
    
class Molecule(MoleculeLike):
    def __init__(self, 
                 name:str,
                 count:int, 
                 translation_rate:float=None, 
                 decay_rate:float=None, 
                 transcription_rate:float=None,
                 transcription_rate_constant:bool=False,
                 k:float=None,
                 c:float=None):
        """
        A molecule object in a cell.

        Parameters:
            transcription_rate: float
                The rate at which the molecule is transcribed
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

    def creation_rate(self, q:int, k:float, c:float) -> float:
        """
        General version of the rate equation for the creation of a molecule from nothing.
        $\new_transcription_rate(Q) = \transcription_rate \frac{q}{q+K}$

        Parameters:
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
        transcription_rate = self.transcription_rate * (q ** c) / (k ** c + q ** c)
        return transcription_rate

    def decay(self, dt:int=None) -> int:
        """
        Decay of a molecule over time
        Parameters:
            dt: int
                The time

        Returns:
            int: The number of molecules left after decay
        """
        # Default time step is 1
        dt = dt or 1
        return self.express(self.decay_rate, dt, from_count=self.count)
    
    def transcription(self, protein:int=None, dt:int=None) -> int:
        """
        Transcription of a molecule over time

        Parameters:
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
            raise ValueError("Protein must be provided for non-constant transcription rate")
        if self.transcription_rate_constant:
            transcription_rate = self.transcription_rate
            from_count = 1
        else:
            transcription_rate = self.creation_rate(protein, self.k, self.c)
            from_count = protein
        return self.express(transcription_rate, dt, from_count=from_count)

    def translation(self, dt:int=None) -> int:
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
    def __init__(self, 
                 name:str,
                 count:int, 
                 molecules_per_complex:List[int], 
                 degradation_rate:int=None, 
                 formation_rate:int=None):
        """
        A complex object in a cell

        Parameters:
            count: int
                The number of complexes
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

    def degradation(self, dt:int=None) -> int:
        """
        Degradation of a molecule over time
        Parameters:
            dt: int
                The time
        
        Returns:
            int: The number of molecules left after degradation
        """
        count_diff = self.express(self.degradation_rate, dt, from_count=self.count)
        return count_diff
    
    def formation(self, molecules:List[int], dt:int=None) -> List[int]:
        """
        Formation of a complex between multiple molecules over time. 
        The complex is formed between multiple molecules 1:1 relationship.
        Parameters:
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
        num_possible_new_complexes = min(np.array(molecules) // np.array(self.molecules_per_complex))
        formed_complexes = self.express(self.formation_rate, dt, from_count=num_possible_new_complexes)
        other_count_change = np.array(self.molecules_per_complex) * formed_complexes
        return formed_complexes, other_count_change

def construct_path(path:str=None, fname:str=None) -> str:
    """
    Construct the path to the file. If the path is not provided, the current working directory is used.

    Parameters:

    """
    path = path or Path.cwd()
    fname = fname or "init_state.yaml"
    fpath = Path(path).joinpath("states", fname)
    if fpath.suffix != ".yaml":
        raise ValueError("File must be a yaml file")
    return fpath

@njit(parallel=True)
def fast_random_occurrence(expression_rate:float, from_count:int) -> np.ndarray:
    """
    Fast random choice for each element in the array. 
    This function is 4 times faster than np.random.choice.

    Parameters:
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