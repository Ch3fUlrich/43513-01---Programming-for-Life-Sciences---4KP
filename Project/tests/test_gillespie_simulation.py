import pytest
import numpy as np
from GillespieSimulation import StateMachine, State, Molecule, Complex


def test_state_machine_initialization():
    """Test basic initialization of Gillespie Simulation"""
    StateMachine()

    assert StateMachine().states
