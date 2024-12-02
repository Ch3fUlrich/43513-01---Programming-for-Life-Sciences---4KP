# defines Molecule related classes

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
    from Project.utils import fast_random_occurrence

    logger.info("Successfully loaded all required libraries.")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    raise ImportError(f"Failed to import required libraries: {e}") from e

class MoleculeLike:
    """
    A base class representing a molecule-like object.

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

    def __init__(self, name: str, count: int) -> None:
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

    def create_molecule_dict(self) -> Dict[str, Union[str, int]]:
        """
        Create a dictionary of the molecule object.

        Returns
        -------
        Dict[str, Union[str, int]]:
            A dictionary containing the molecule's attributes.
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
        Simulate the expression or decay of a molecule over time.

        Random choice for every molecule.

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
        dt = dt or 1
        if from_count is None:
            from_count = self.count

        if from_count < 0:  # Log and correct if from_count is negative
            logger.warning(f"Negative count detected for {self.name}. Correcting to 0.")
            from_count = 0

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
        translation_rate: Optional[float] = None,
        decay_rate: Optional[float] = None,
        transcription_rate: Optional[float] = None,
        transcription_rate_constant: bool = False,
        k: Optional[float] = None,
        c: Optional[float] = None,
    ) -> None:
        """
        Initialize a molecule object with its attributes.

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
        Calculate the rate of molecule creation.

        Uses generalized Hill function.
        This method computes the creation rate of a molecule based on the
        number of molecules (`q`), the half-maximal activation constant (`k`),
        and the steepness
        of the activation curve controlled by the Hill coefficient (`c`).
        The equation is given by:

        new_transcription_rate(Q) = transcription_rate * (q^c / (q^c + k^c))

        Parameters
        ----------
        q: int
            The number of TF molecules
        k: float
            Half-maximal activation constant
        c: float
            The Hill coefficient, controlling the steepness of the
            activation curve.
            Default is 1, which corresponds to linear activation.
        Returns
        -------
        float:
            The new creation rate of the molecule
        """
        c = c or 1
        transcription_rate = self.transcription_rate * (q**c) / (k**c + q**c)
        return transcription_rate

    def decay(self, dt: Optional[int] = None) -> int:
        """
        Decay of a molecule over time.

        Parameters:
        dt: Optional[int]
            The time step


        Returns
        -------
        int:
            The number of molecules left after decay
        """
        dt = dt or 1
        decayed_molecules = self.express(self.decay_rate, dt, from_count=self.count)
        if decayed_molecules < 0:
            raise ValueError("Negative count after decay")
        elif decayed_molecules > self.count:
            raise ValueError("Decayed molecules exceed initial count")
        return decayed_molecules

    def transcription(
        self,
        protein: Optional[int] = None,
        dt: Optional[int] = None,
    ) -> int:
        """
        Simulate molecule transcription over time.

        Parameters
        ----------
        protein: int, optional
            The number of protein molecules influencing transcription.
            If the transcription rate is
            non-constant, this value must be provided.
        dt: int
            The time step over which transcription is
            simulated. Defaults to 1 if not provided.

        Returns
        -------
        int:
            The number of molecules transcribed.

        Raises
        ------
        ValueError
            If `protein` is not provided and the
            transcription rate is not constant.
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
        Simulate molecule translation over time.

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
        degradation_rate: Optional[int] = None,
        formation_rate: Optional[int] = None,
    ) -> None:
        """
        Initialize a complex object with its attributes.

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

    def degradation(self, dt: Optional[int] = None) -> int:
        """
        Simulate degradation of a molecule over time.

        Parameters
        ----------
        dt: int, optional
            The time step for the degradation simulation. Default is 1.

        Returns
        -------
        int:
            The number of molecules degraded.
            #### clarify whether the return value represents the
            # number of molecules degraded or remaining
        """
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
        Simulate the formation of a complex over time.

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
