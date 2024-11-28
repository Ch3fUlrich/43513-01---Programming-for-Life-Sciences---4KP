import pytest
import numpy as np
from Project.Gillespie_Simulation import StateMachine, State, Molecule, Complex


def test_state_machine_initialization():
    """Test basic initialization of Gillespie Simulation"""
    sim = StateMachine()
