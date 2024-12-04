# 43513-01 – Programming for Life Sciences - 4KP <img src="https://pfst.cf2.poecdn.net/base/image/b199686416c40d6c29f5cb5ea7c87d8aa11c57c95fce53febead730a140ab9bd?w=1024&h=768&pmaid=166853149)" alt="gene" width="70"/>

This Repository is mandatory for the course mentioned in the title. Weekly lectures are added to the repository as well as the progress in the project that is used for passing the lecture.

**The Project will be solved using Jupyter Notebook and Python 3.11**

# Gillespie Simulation Project
## Synopsis
This project implements a Gillespie simulation of a microRNA-transcription factor-target protein feed-forward loop (FFL) in gene regulation. This stochastic simulation algorithm is used to model the time evolution of well-mixed biochemical systems.

## Usage
### Define Initial Parameters
Before running the simulation, the initial parameters of the molecules that govern the dynamics of this biological system are defined and loaded. These parameters represent the starting state, rates (e.g., transcription, degradation, and formation, etc.), and interactions of the key molecular species in the system. 

We have set a biologically meaningful and computationally stable state, which can be found in `./Project/states/initial_state.yaml`.

Defining such states before each run of the simulation is crucial for simulation consistency and it ensures that the model produces accurate and reliable results.
 
 **IMPORTANT:** The naming must be the same as in the example file.

### Parameters Description 
Please refer to `./Images/miRNA-interference_scheme.png` for a schematic of the FFL network being simulated.

**Initial Time (time)**
The simulation begins at time 0. Time is used to track the progression of events during the simulation.

**Transcription Factor mRNA (TF_mRNA, w)**
TF_mRNA is the precursor for the transcription factor protein (TF_protein). Its levels are governed by the rates of its synthesis (via transcription), its decay, and its translation into protein.

**Transcription Factor Protein (TF_protein, q)**
TF_protein regulates the transcription of both miRNA (s) and mRNA (r). Its count and decay rate directly affect the system's regulatory feedback loops.

**miRNA (s)**
miRNA regulates mRNA (r) through complex formation, influencing target protein (p) expression. Its transcription and decay rates define the strength and duration of its regulatory role.

**mRNA (r)**
mRNA encodes for the target protein (p). Its levels directly affect protein levels and are goverened by the rates of its synthesis, decay, and translation.

**protein (p)**
The target protein is the final output of the network. Its decay rate determines how quickly its levels respond to changesin regulatory inputs.

**miRNA-mRNA Complex (g)**
The complex results from the interaction between miRNA (s) and mRNA (r), which represses mRNA's translation into protein. The complex also dissociates back into miRNA and mRNA

### Run
To run the simulation:
```console
cd Project
python Gillespie_Simulation.py
```
or you provide the initial state (.yaml file):
``` console
cd Project
python Gillespie_Simulation.py -i init_state.yaml
```
For additional information, run
```console
python Gillespie_Simulation.py -h
```
## Installation
### Repo
To download this repo, simply run:
```bash
git clone https://github.com/Ch3fUlrich/43513-01---Programming-for-Life-Sciences---4KP.git
```
After that, the required python packages must be installed before running the codes.
### Requirements
#### Option 1: pip
One way to install all project dependencies is using pip:
```bash
pip install -r requirements.txt
# install own package
pip install -e .
# for development
pip install -r requirements_dev.txt
```
#### Option 2: conda environment
Alternatively, a conda environment can be created, and all dependencies installed automatically using:
```bash
conda env create -f environment.yml
```
<strong>Note:</strong> if a conda environment is created, be sure to activate it before running the codes, using:

```bash
conda activate gillespie
```
## Background
Biological processes are often stochastic in nature. This simulation focuses on a genetic circuit that can buffer stochastic fluctuations in gene expression. The circuit consists of:
- A transcriptional regulator (transcription factor, q)
- A post-transcriptional regulator (microRNA, s)
- A target protein-coding gene (p)
The transcription factor activates the target gene, while the microRNA induces degradation of the target mRNA, repressing its expression (incoherent feed-forward loop). More details can be found in the [Background Information](background.md).
## Reactions
The simulation includes the following reactions:
1. TF transcription and decay
2. TF translation and protein decay
3. miRNA production and decay
4. Target mRNA transcription and decay
5. Target protein translation and decay
6. miRNA-mediated mRNA degradation
## Gillespie Algorithm
The Gillespie algorithm is used to simulate the stochastic dynamics of the system. It involves:
1. Calculating reaction rates based on current molecule numbers
2. Determining the time to the next reaction
3. Selecting which reaction occurs
4. Updating molecular species and time

## Versioning 
This is a student project, where each of the students contributes via their own branch and then pushes a merge request to the main branch.
## Contributors
Students of the course "Programming for Life Sciences" at the Biozentrum Basel.

## License
This project is not yet licensed.
