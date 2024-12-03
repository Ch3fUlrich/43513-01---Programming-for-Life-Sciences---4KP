import pytest
import numpy as np
from Project.classes.State import StateMachine, State


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


def test_state_initialization(simulator):
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


if __name__ == "__main__":
    test_state_machine_initialization()
    test_state_next_state_calculation()
