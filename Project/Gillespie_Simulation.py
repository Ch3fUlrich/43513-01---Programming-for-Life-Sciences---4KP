# type hints
from typing import List

# calculation
import numpy as np

# plotting
import matplotlib.pyplot as plt
# show time till finished trange
from tqdm import trange

# loading files
import yaml
from pathlib import Path

class State_Machine:
    def __init__(self, innit_state_path:str, dt:int=1):
        """
        A state machine object in a cell.

        Parameters:
            state: State
                The state of the cell
            dt: int
                The time step
        """
        self.path = Path(innit_state_path).parent
        self.state = State(innit_state_path)
        self.dt = dt
        self.times = None
        self.molecule_counts = None

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
        fname = fname+".npy" or "molecule_counts.npy"
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
        molecule_counts = np.array((trajectories, steps+1, n_molecules))
        times = np.zeros((trajectories, steps+1))
        for i in trange(trajectories):
            for j in trange(steps):
                self.state.next_state(self.dt)
                times[i, j] = self.state.time
                molecule_counts[i, j] = self.state.extract_molecule_counts()
        if save:
            if not saven_fname:
                saven_fname = f"molecule_counts_{steps}_{trajectories}.npy"
            self.save_runs(molecule_counts, fname=saven_fname, path=save_path)
        return molecule_counts

    def plot(self):
        """
        Plot the state of the cell.
        """
        plt.plot(self.times, self.molecule_counts)
        plt.show()
        raise NotImplementedError("optimie the plot method")

class State:
    def __init__(self, path:str):
        """
        A state object in a cell.

        Parameters:
            path: str
                The path of the state yaml file
            molecules: dict

        """
        if not isinstance(path, str) or not isinstance(path, Path):
            raise ValueError("Path must be a string or Path object")
        assert Path(path).suffix == '.yaml', "File must be a yaml file"

        self.path = Path(path)
        self.state_dict = self.load_state()
        self.time = self.state_dict["time"]
        self.molecules:List = self.create_molecules()

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
        for molecule in self.molecules:
            molecule_dict = molecule.create_molecule_dict()
            state_dict.update(molecule_dict)
        if save:
            self.save_state()
        return self.state_dict

    def create_molecules(self):
        """
        Create the molecule objects in the cell.
        """
        molecules = []
        for key, value in self.state_dict.items():
            if key != "time":
                molecules.append(Molecule(value))
                #TODO: molecule object creation need to be automated
                raise ValueError("Molecule type not recognized")
        return molecules

    def next_state(self, dt:int=1):
        """
        Update the state of the cell.

        Parameters:
            dt: int
                The time step
        """
        #TODO: Implement the next state method
        ...
        raise NotImplementedError("Subclasses must implement next_state method")
    
    def extract_molecule_counts(self) -> List[int]:
        """
        Extract the number of molecules in the cell.

        Returns:
            List[int]: The number of molecules in the cell
        """
        counts = [molecule.molecule for molecule in self.molecules]
        return counts

class MoleculeLike:
    def __init__(self, name:str, molecule:int):
        """
        A molecule like object in a cell
        Parameters:
            molecule: int
                The number of molecules
        """ 
        self.name = name
        self.molecule = molecule

    def __add__(self, molecule_change:int) -> int:
        """
        Add the number of molecules. Overloading the + operator.

        Returns:
            int: The number of molecules left after addition
        """
        self.molecule += molecule_change
        return self.molecule
    
    def __sub__(self, molecule_change:int) -> int:
        """
        Subtract the number of molecules. Overloading the - operator.

        Returns:
            int: The number of molecules left after subtraction
        """
        self.molecule -= molecule_change
        return self.molecule
    
    def create_molecule_dict(self) -> dict:
        """
        Create a dictionary of the molecule object
        """
        molecule_dict = {}
        for key, value in self.__dict__.items():
            if value:
                molecule_dict[key] = value
        return molecule_dict

    def express(self, expression_rate:int, dt:int=None) -> int:
        """
        Expression of a molecule over time
        Parameters:
            expression_rate: int
                The rate at which the molecule is expressed
            dt: int
                The time
        
        Returns:
            int: The number of molecules left after expression
        """
        # Default time step is 1
        dt = dt or 1
        other_molecule = self.molecule * expression_rate * dt
        return other_molecule
    
class Molecule(MoleculeLike):
    def __init__(self, molecule:int, 
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
        super().__init__(molecule)
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
        self.transcription_rate = self.transcription_rate * (q ** c) / (k ** c + q ** c)
        return self.transcription_rate

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
        self.molecule = self.express(-self.decay_rate, dt)
        return self.molecule
    
    def transcription(self, protein:int, dt:int=None) -> int:
        """
        Transcription of a molecule over time

        Parameters:
            protein: int
                The number of proteins
            dt: int
                The time
        
        Returns:
            int: The number of molecules left after transcription
        """
        transcription_rate = self.transcription_rate if self.transcription_rate_constant else self.creation_rate(protein, self.k, self.c)
        return self.express(transcription_rate, dt)

    def translation(self, dt:int=None) -> int:
        """
        Translation of a molecule over time

        Parameters:
            dt: int
                The time

        Returns:
            int: The number of molecules left after translation
        """
        return self.express(self.translation_rate, dt)

class Complex(MoleculeLike):
    def __init__(self, num_complex:int, molecules_per_complex:List[int], degradation_rate:int=None, formation_rate:int=None):
        """
        A complex object in a cell

        Parameters:
            num_complex: int
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
        super().__init__(num_complex)
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
        self.molecule = self.express(-self.degradation_rate, dt)
        other_molecule = self.molecule
        return self.molecule, other_molecule
    
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
        num_possible_new_complexes = min([molecule // self.molecules_per_complex for molecule in molecules])
        formed_complexes = num_possible_new_complexes * self.formation_rate * dt
        self.molecule += formed_complexes
        other_molecules = np.array(molecules) - np.array(self.molecules_per_complex) * formed_complexes
        return self.molecule, other_molecules