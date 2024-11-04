# 43513-01 – Programming for Life Sciences - 4KP <img src="https://pfst.cf2.poecdn.net/base/image/b199686416c40d6c29f5cb5ea7c87d8aa11c57c95fce53febead730a140ab9bd?w=1024&h=768&pmaid=166853149)" alt="gene" width="70"/>

This Repository is mandatory for the course mentioned in the title. Weekly lectures are added to the repository as well as the progress in the project that is used for passing the lecture.

**The Project will be solved using Jupyter Notebook and Python 3.11**

# TODO
- [x] Split Gillespie Example into functions for easier understanding
- [x] Transform Project Goals into easy-to-understand formulas [Better Description](#better-description)
- [x] Look into [Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001101) explaining the project
  - [x] extract constants from paper
- [ ] Implement functions for rules found in [Better Description](#better-description)
  - [ ] create Classes with functions
    - [x] MoleculeLike
      - [x] General expression function for functions below
      - [x] add molecules using operator overloading
      - [x] substract molecules using operator overloading
      - [x] general creation_rate function for molecules created from nothing
    - [x] Molecule
      - [x] transcription
      - [x] translation
      - [x] decay
    - [x] Complex
      - [x] complex degradation
      - [x] complex formation
    - [x] State - compost of molecules representing the world
      - [x] next State - Glippsie version of state change
    - [x] State Machine - tracking states over time and trajectories
  - [ ] Plotting
    - [ ] Plotting function for single run
    - [x] Plotting function for the results of the Gillespie function
- [ ] Write tests for automated testing
- [ ] Add Logging


# Gillespie Simulation Project
## Synopsis
This project implements a Gillespie simulation of a microRNA-transcription factor-target protein feed-forward loop (FFL) in gene regulation. This stochastic simulation algorithm is used to model the time evolution of well-mixed biochemical systems.
## Installation
### Repo
To download this repo, simply run:
```console
git clone https://github.com/Ch3fUlrich/43513-01---Programming-for-Life-Sciences---4KP.git
```
After that, the required python packages must be installed before running the codes.
### Requirements
#### Option 1: pip
One way to install all project dependencies is using pip:
```console
pip install -r requirements.txt
```
#### Option 2: conda environment
Alternatively, a conda environment can be created, and all dependencies installed automatically using:
```console
conda env create -f environment.yml
```
<strong>Note:</strong> if a conda environment is created, be sure to activate it before running the codes, using:
```console
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
