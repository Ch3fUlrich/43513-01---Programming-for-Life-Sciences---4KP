import numpy as np
import matplotlib.pyplot as plt

def gillespi_simulation(N_A:int, 
                        N_B:int, 
                        N_AB:int, 
                        start_T:int=0, 
                        rf:float=0.1,
                        rb:float=1.0,
                        steps:int=100):
    """
    Run a Gillespi simulation for multiple steps.

    Parameters:
    N_A: int
        Initial number of A molecules
    N_B: int
        Initial number of B molecules
    N_AB: int
        Initial number of AB molecules
    T: int
        Time of simulation
    steps: int
        Number of steps    
    """
    # Set up data arrays
    N_A_steps = np.zeros(steps+1)
    N_B_steps = np.zeros(steps+1)
    N_AB_steps = np.zeros(steps+1)
    T_steps = np.zeros(steps+1)

    # Set initial conditions
    N_A_steps[0] = N_A
    N_B_steps[0] = N_B
    N_AB_steps[0] = N_AB
    T_steps[0] = start_T

    for i in range(steps):
        N_A = N_A_steps[i]
        N_B = N_B_steps[i]
        N_AB = N_AB_steps[i]

        N_A_next, N_B_next, N_AB_next, dt = gillespi_simulation_step(N_A, N_B, N_AB, rf, rb)

        # Update the data arrays
        N_A_steps[i+1] = N_A_next
        N_B_steps[i+1] = N_B_next
        N_AB_steps[i+1] = N_AB_next
        T_steps[i+1] = T_steps[i] + dt        

    return N_A_steps, N_B_steps, N_AB_steps, T_steps

def gillespi_simulation_step(N_A:int, N_B:int, N_AB:int, rf:float, rb:float):
    """
    Run a single step of the Gillespi simulation.
    """
    # Calculate the reaction rates
    R = calc_reaction_rate(N_A, N_B, N_AB, rf, rb)

    # Calculate time to next reaction
    dt = calc_time_to_next_reaction(R)
    
    # Select next reaction
    pf = calc_forward_reaction_probability(N_A, N_B, N_AB, rf, R)
    u = np.random.random()

    # Apply the reaction
    if u < pf:
        N_A_next, N_B_next, N_AB_next = forward_reaction(N_A, N_B, N_AB)
    else:
        N_A_next, N_B_next, N_AB_next = backward_reaction(N_A, N_B, N_AB)

    return N_A_next, N_B_next, N_AB_next, dt

def forward_reaction(N_A:int, N_B:int, N_AB:int):
    """
    Apply the forward reaction.
    """
    N_A -= 1
    N_B -= 1
    N_AB += 1

    return N_A, N_B, N_AB

def backward_reaction(N_A:int, N_B:int, N_AB:int):
    """
    Apply the backward reaction.
    """
    N_A += 1
    N_B += 1
    N_AB -= 1

    return N_A, N_B, N_AB

def calc_forward_reaction_probability(N_A:int, N_B:int, N_AB:int, rf:float, R:float):
    """
    Calculate the probability of the forward reaction.
    """
    return rf * N_A * N_B / R

def calc_reaction_rate(N_A:int, N_B:int, N_AB:int, rf:float, rb:float):
    """
    Calculate the reaction rate for the given system.
    """
    return rf * N_A * N_B + rb * N_AB

def calc_time_to_next_reaction(R:float):
    """
    Calculate the time to the next reaction.
    Code in this function is the same as running np.random.exponential(1/R)
    """
    u = np.random.random() # same as np.random.uniform(0,1)
    return 1/R * np.log(1/u)

def multi_gillespi_simulation(N_A:int,
                                N_B:int,
                                N_AB:int,
                                start_T:int=0,
                                rf:float=0.1,
                                rb:float=1.0,
                                steps:int=100,
                                trajectories:int=100):
    """
    Run multiple Gillespi simulations and return the results.

    Parameters:
    N_A: int
        Initial number of A molecules
    N_B: int
        Initial number of B molecules
    N_AB: int
        Initial number of AB molecules
    T: int
        Time of simulation
    steps: int
        Number of steps
    trajectories: int

    Returns:
    multi_N_A: np.ndarray
        Array of N_A values for each trajectory
    multi_N_B: np.ndarray
        Array of N_B values for each trajectory
    multi_N_AB: np.ndarray
        Array of N_AB values for each trajectory
    """
    # Set up data arrays
    multi_N_A = np.zeros((trajectories, steps+1))
    multi_N_B = np.zeros((trajectories, steps+1))
    multi_N_AB = np.zeros((trajectories, steps+1))
    multi_T = np.zeros((trajectories, steps+1))

    for i in range(trajectories):
        N_A_trajectory, N_B_trajectory, N_AB_trajectory, T_trajectory = gillespi_simulation(N_A, N_B, N_AB, start_T, rf, rb, steps)
        
        # Update the data arrays
        multi_N_A[i] = N_A_trajectory
        multi_N_B[i] = N_B_trajectory
        multi_N_AB[i] = N_AB_trajectory
        multi_T[i] = T_trajectory

    return multi_N_A, multi_N_B, multi_N_AB, multi_T

def plot_simulation_results(N_A: np.ndarray, 
                            N_B: np.ndarray, 
                            N_AB: np.ndarray, 
                            T: np.ndarray,
                            molecule_type = ['A', 'B', "AB"],
                            figsize=(7,15)
                            ):
    """
    Plot the results of the Gillespi simulation.
    """
    # Plot the simulated trajectories for each molecule type
    fig, axs = plt.subplots(3, 1, figsize=figsize)
    num_trajectories = N_A.shape[0]

    molecule_type = ['A', 'B', "AB"]
    for i in range(3):
        axs[i].set_title(f'Number of {molecule_type[i]} molecules')
        axs[i].set_xlabel("Time (hours)")
        axs[i].set_ylabel("Counts")
        
    # Plot each simulated trajectory
    for i in range(num_trajectories):
        axs[0].plot(T[i,:], N_A[i,:], marker='', color='red', linewidth=0.6, alpha=0.3)
        axs[1].plot(T[i,:], N_B[i,:], marker='', color='green', linewidth=0.6, alpha=0.3)
        axs[2].plot(T[i,:], N_AB[i,:], marker='', color='black', linewidth=0.6, alpha=0.3)

    plt.show()

def plot_simulation_examples(N_A: np.ndarray, 
                            N_B: np.ndarray, 
                            N_AB: np.ndarray, 
                            T: np.ndarray,
                            molecule_type = ['A', 'B', "AB"],
                            figsize=(7,15)
                            ):
    """
    Plot a few examples of the simulated trajectories.
    """
    # Let's also plot a few simulations
    num_trajectories = N_A.shape[0]

    n2plot = 3
    is2plot = np.random.choice(list(range(num_trajectories)), size=n2plot, replace=False)
    fig, axs = plt.subplots(n2plot, 1, figsize=figsize)

    for i in range(n2plot):
        axs[i].set_title(f'Number of molecules for simulation {is2plot[i]}')
        axs[i].set_xlabel("Time (hours)")
        axs[i].set_ylabel("Counts")
        axs[i].plot(T[i,:], N_A[i,:], marker='', color='red', linewidth=2.0, alpha=0.7)
        axs[i].plot(T[i,:], N_B[i,:], marker='', color='blue', linewidth=2.0, alpha=0.7)
        axs[i].plot(T[i,:], N_AB[i,:], marker='', color='black', linewidth=2.0, alpha=0.7)
        axs[i].legend(molecule_type)
    plt.show()


def example_setting():
    """
    # Reactions:
    # A + B -> AB
    # AB -> A + B

    # Molecular counts
    # N_A
    # N_B
    # N_AB

    # Reaction rates
    # rf : reaction rate for A + B -> AB
    # rb : reaction rate for AB -> A + B
    """
    # Set reaction rates 
    rf = 0.1        # Forward reaction rate
    rb = 1.0        # Backwards reaction rate

    # Set initial conditions
    T = 0 # for completeness
    N_A = 10
    N_B = 30
    N_AB = 0

    # Set meta-parameters
    steps = 100
    trajectories = 100

    multi_N_A, multi_N_B, multi_N_AB, multi_T = multi_gillespi_simulation(N_A, 
                                                                 N_B, 
                                                                 N_AB,
                                                                 T, 
                                                                 rf,
                                                                 rb,
                                                                 steps, 
                                                                 trajectories)

    plot_simulation_results(multi_N_A, multi_N_B, multi_N_AB, multi_T)
    plot_simulation_examples(multi_N_A, multi_N_B, multi_N_AB, multi_T)

