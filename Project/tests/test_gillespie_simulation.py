import pytest
import numpy as np
from Project.classes.State import StateMachine, State
from Project.classes.Molecule import Molecule, Complex
from pathlib import Path
import os


@pytest.fixture
def simulator():
    """Fixture to create a StateMachine instance for testing"""
    return StateMachine()


def test_state_machine_initialization():
    """Test basic initialization of Gillespie Simulation"""
    simulator = StateMachine()

    assert isinstance(simulator, StateMachine)
    assert isinstance(simulator.state, State)
    assert isinstance(simulator.molecule_counts, type(None))
    assert isinstance(simulator.times, type(None))

    with pytest.raises(TypeError):
        StateMachine(init_state_path=1234)
    with pytest.raises(ValueError):
        StateMachine(init_state_path="bad_path")
        StateMachine(init_state_path="init_state_path.txt")
        StateMachine(init_state_path="init_state_path.txt", state=simulator.state)

    test_state_initialization(simulator)


def test_state_initialization(simulator: StateMachine):
    """Test basic initialization of State"""
    state = simulator.state
    state_summary = state.print()
    state_full_description = state.print(full=True)

    assert isinstance(state_summary, str)
    assert isinstance(state_full_description, str)


def test_state_next_state_calculation():
    """Test next state calculation"""
    simulator = StateMachine()
    state = simulator.state

    for i in range(100):
        state.next_state()
        state.set_init_state()


@pytest.mark.parametrize("count, decay_rate, dt, expected_decay, tolerance", [
    (100, 0.1, 1, 10, 2),   # Allow ±2 tolerance
    (200, 0.05, 2, 20, 4),  # Allow ±4 tolerance
    (50, 0.2, 0, 0, 0),     # No decay for dt=0
    (0, 0.1, 1, 0, 0),      # No decay when count=0
    (100, 0.0, 1, 0, 0),    # No decay when decay_rate=0
    (100, 1.0, 1, 100, 0),  # All molecules decay when decay_rate=1.0
])
def test_decay_calculation(count, decay_rate, dt, expected_decay, tolerance):
    molecule = Molecule(name="test_molecule", count=count, decay_rate=decay_rate)
    decayed_count = molecule.decay(dt=dt)
    assert isinstance(decayed_count, int), "Decay count should be an integer."
    assert 0 <= decayed_count <= count, "Decay count should be within valid range."
    assert expected_decay - tolerance <= decayed_count <= expected_decay + tolerance, "Decay calculation is incorrect."


def test_express_calculation():
    """Test the express calculation for a molecule."""
    molecule = Molecule(name="test_molecule", count=100, translation_rate=0.2)

    # Simulate expression over one timestep
    expressed_count = molecule.express(expression_rate=0.2, dt=1, from_count=100)

    # Ensure the expressed count is a valid integer and does not exceed the original count
    assert isinstance(expressed_count, int), "Expressed count should be an integer."
    assert 0 <= expressed_count <= molecule.count, "Expressed count should be within valid range."


@pytest.mark.parametrize("from_count, expression_rate, dt, expected_expressed", [
    (100, 0.2, 1, 20),   # Typical case
    (50, 0.1, 2, 10),    # Expression over 2 timesteps
    (0, 0.2, 1, 0),      # No expression when from_count is 0
    (100, 0.0, 1, 0),    # No expression when expression_rate is 0
    (100, 0.5, 1, 50),   # High expression rate
])
def test_express_calculation(from_count, expression_rate, dt, expected_expressed):
    """Test the express calculation for a molecule."""
    molecule = Molecule(name="test_molecule", count=100, translation_rate=0.2)   
    # Simulate expression
    expressed_count = molecule.express(expression_rate=expression_rate, dt=dt, from_count=from_count)
    # Validate results
    assert isinstance(expressed_count, int), "Expressed count should be an integer."
    assert 0 <= expressed_count <= from_count, "Expressed count should be within valid range."
    assert expressed_count == expected_expressed, "Express calculation is incorrect."


if __name__ == "__main__":
    print("test-------------------")
    test_state_machine_initialization()
    test_state_next_state_calculation()
    test_decay_calculation()
    test_express_calculation()
