import pytest
import numpy as np
from Project.GillespieSimulation import StateMachine, State, Molecule, Complex
from pathlib import Path

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

def test_plot_creation(tmp_path):
    """
    Test that the simulation graph is created.
    Ensures that the `plot` method generates and saves a plot
    file when the simulation is run.
    """
    # Setup: Create a temporary directory for outputs
    output_folder = tmp_path / "output"
    output_folder.mkdir()

    # Initialize a mock state and state machine
    init_state_path = Path("tests/mock_data/initial_state.yaml")  # Replace with a valid mock path
    state = State(init_state_path)
    simulator = StateMachine(state=state)

    # Run the simulation
    simulator.run(steps=10, trajectories=5, save=False)  # Small-scale simulation

    # Generate the plot
    simulator.plot(save_folder=str(output_folder))

    # Check that the plot file was created
    plot_file = output_folder / "simulation_plot.png"
    assert plot_file.exists(), f"Plot file was not created: {plot_file}"

    # Optional: Validate that the plot file is non-empty
    assert plot_file.stat().st_size > 0, f"Plot file is empty: {plot_file}"


def test_decay_calculation():
    """Test the decay calculation for a molecule."""
    molecule = Molecule(name="test_molecule", count=100, decay_rate=0.1)

    # Simulate decay over one timestep
    decayed_count = molecule.decay(dt=1)

    # Ensure the decay count is a valid integer and does not exceed the original count
    assert isinstance(decayed_count, int), "Decay count should be an integer."
    assert 0 <= decayed_count <= molecule.count, "Decay count should be within valid range."

def test_complex_creation():
    """Test the formation of a molecular complex."""
    complex_molecule = Complex(
        name="test_complex",
        count=0,
        molecules_per_complex=[2, 3],
        formation_rate=0.2,
        degradation_rate=0.1
    )

    # Simulate complex formation with available molecules
    molecules = [10, 15]  # Counts of the two types of molecules required for complex formation
    formed_complexes, used_molecules = complex_molecule.formation(molecules, dt=1)

    # Ensure the formed complexes count is valid
    assert isinstance(formed_complexes, int), "Formed complexes count should be an integer."
    assert 0 <= formed_complexes <= min(molecules[0] // 2, molecules[1] // 3), "Formed complexes count is out of bounds."

    # Ensure the correct number of molecules were used
    assert len(used_molecules) == 2, "Used molecules list should have two elements."
    assert all(isinstance(x, int) for x in used_molecules), "All used molecules counts should be integers."
    assert all(x >= 0 for x in used_molecules), "Used molecules counts should be non-negative."

def test_express_calculation():
    """Test the express calculation for a molecule."""
    molecule = Molecule(name="test_molecule", count=100, translation_rate=0.2)

    # Simulate expression over one timestep
    expressed_count = molecule.express(expression_rate=0.2, dt=1, from_count=100)

    # Ensure the expressed count is a valid integer and does not exceed the original count
    assert isinstance(expressed_count, int), "Expressed count should be an integer."
    assert 0 <= expressed_count <= molecule.count, "Expressed count should be within valid range."


if __name__ == "__main__":
    test_state_machine_initialization()
    test_state_next_state_calculation()
    test_decay_calculation()
    test_complex_creation()
    test_express_calculation()
    pytest.main()
