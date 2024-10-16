from typing import List
import numpy as np
import matplotlib.pyplot as plt

def incoherent_feed_forward_loop():
    """
    An incoherent feed-forward loop can lead to dynamics, such as pulse-like responses or resopnse acceleration, which can help in noise reduction

    Feed-forward loop: 
        The system is a feed-forward loop because the transcription factor (TF) regulates
        - miRNA and
        - target gene, 
        creating a path that "feeds forward" from the TF to the target gene through two different routes.

    Incoherent (Opposing effects):
        - Positive: TF activates
            - directly the transcription of target mRNA. 
        - Negative: 
            - indirect negative regulation by production of miRNA, which in turn represses (degrading) the target mRNA
    """

    raise NotImplementedError("Implement the incoherent feed-forward loop")


class MoleculeLike:
    def __init__(self, molecule:int, 
                 transcription_rate:float=None, 
                 translation_rate:float=None, 
                 decay_rate:float=None,
                 alpha:float=None):
        """
        A molecule like object in the cell
        Parameters:
            molecule: int
                The number of molecules
            transcription_rate: float
                The rate at which the molecule is transcribed
            translation_rate: float
                The rate at which the molecule is translated
            decay_rate: float
                The rate at which the molecule decays
            alpha: float
                The rate at which the molecule is created
        """ 
        self.molecule = molecule
        self.translation_rate = translation_rate
        self.transcription_rate = transcription_rate
        self.decay_rate = decay_rate
        self.alpha = alpha

    def __add__(self, molecule_change:int) -> int:
        """
        Add the number of molecules
        """
        self.molecule += molecule_change
        return self.molecule
    
    def __sub__(self, molecule_change:int) -> int:
        """
        Subtract the number of molecules
        """
        self.molecule -= molecule_change
        return self.molecule
    
    def express(self, expression_rate:int, dt:int=None) -> int:
        """
        Expression of a molecule over time
        Parameters:
            expression_rate: int
                The rate at which the molecule is expressed
            dt: int
                The time
        """
        # Default time step is 1
        dt = dt or 1
        other_molecule = self.molecule * expression_rate * dt
        return other_molecule

    def creation_rate(self, q:int, k:float, c:float=None) -> float:
        """
        General version of the rate equation for the creation of a molecule from nothing.
        $\new_alpha(Q) = \alpha \frac{q}{q+K}$

        Parameters:
            q: int
                The number of TF molecules
            k: float
                Half-maximal activation constant
            c: float
                Hill coefficient for fixing steepness of the activation curve. Default value is 1 for linear activation
        """
        # Hill coefficient set to 1 for linear activation if not provided
        c = c or 1
        self.alpha = self.alpha * (q ** c) / (k ** c + q ** c)
        return self.alpha
    
class Molecule(MoleculeLike):
    def __init__(self, molecule:int, transcription_rate:float=None, translation_rate:float=None, decay_rate:float=None):
        super().__init__(molecule, transcription_rate, translation_rate, decay_rate)
    
    def decay(self, dt:int=None) -> int:
        """
        Decay of a molecule over time
        Parameters:
            dt: int
                The time
        """
        # Default time step is 1
        dt = dt or 1
        self.molecule = self.express(-self.decay_rate, dt)
        return self.molecule
    
    def transcription(self, dt:int=None) -> int:
        """
        Transcription of a molecule over time
        """
        return self.express(self.transcription_rate, dt)

    def translation(self, dt:int=None) -> int:
        """
        Translation of a molecule over time
        """
        return self.express(self.translation_rate, dt)


class Complex(MoleculeLike):
    def __init__(self, molecule:int):
        self.molecule = molecule

    def degradation(self, degradation_rate:int, dt:int=None) -> int:
        """
        Degradation of a molecule over time
        """
        self.molecule = self.express(-degradation_rate, dt)
        other_molecule = self.molecule
        return self.molecule, other_molecule
    
    def formation(self, molecules:List[int], complex_rate:int, num_molecules:List[int]=None, dt:int=None) -> List[int]:
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
        """
        # Default time step is 1
        dt = dt or 1
        # Default number of molecules is 1 for each molecule
        #TODO: this is probably wrong
        num_molecules = num_molecules or [1]*len(molecules)
        num_molecules *= complex_rate * dt
        new_molecules = np.array(molecules) - np.array(num_molecules)
        other_molecule = molecules[0] - new_molecules[0]
        return new_molecules, other_molecule