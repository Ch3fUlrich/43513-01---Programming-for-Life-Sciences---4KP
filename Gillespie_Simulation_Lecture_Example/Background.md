## Project (Gillespie Simulation)
Stochastic chemical reactions.
The Gillespie algorithm is a stochastic simulation algorithm used to simulate the time evolution of a well-mixed system of chemical species. It is often used in the fields of systems biology and biochemistry. The algorithm is named after Daniel T. Gillespie.

### Background
Biological processes are often stochastic in nature. Small RNAs known as [microRNAs](https://www.nature.com/articles/35002607) (miRNAs (```s```)) were implicated in [developmental robustness](https://www.nature.com/articles/ng1803) by means of suppressing noise in gene regulatory networks. The genetic circuit that can buffer stochastic fluctuations in gene expression. This circuit is composed of a transcriptional regulator (transcription factor, ```q```), a post-transcriptional regulator (miRNA (```s```)) and a target (protein-coding) gene for protein ```p```. The ```q``` activates the target gene, and the miRNA (```s```) induces the degradation of the protein-coding gene's mRNA ```r```, thereby repressing its expression (**incoherent feed-forward loop**). Its dynamics has been [analyzed](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001101).

<img src="https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1001101.g001" alt="regulation_explain1" width="270"/>

#### Example
The following example illustrates the stochastic dynamics of a simple gene regulatory network. The network consists of a single gene that can be in one of two states: active (A) or inactive (I). The gene can switch between these two states by means of two reactions: activation and inactivation. The activation reaction is catalyzed by a transcription factor (```q```) and the inactivation reaction is spontaneous.

<img src="https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1001101.g002" alt="regulation_explain2" width="300"/>
<img src="https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1001101.g003" alt="regulation_explain3" width="300"/>

- $s$ is the miRNA, 
- $q$ is the transcription factor protein, 
- $r$ is the target mRNA, 
- $p$ is the target protein.

The $y$-axis of the plots shows the fluctuations of various quantities around their respective means, in two regulatory scenarios, namely the incoherent FFL (iFFL, top) and the linear TF-target network. In the linear network, the target mRNA $r$ and protein $p$ follow reasonably well the fluctuations in the transcription factor $q$, which induces their expression. In the iFFl, the miRNA $s$ and target mRNA $r$ also follow the TF, while the target protein does not. Furthermore, the fluctuations in the target protein level $q$ are smaller when the target is regulated in an iFFL (top plot) compared to being regulated solely at the level of transcription (bottom plot).


### Goal
Write the code for a Gillespie simulation of the miRNA-TF-target protein FFL. The reactions you will need to include are:
- TF transcription (at a constant rate $\alpha_T$)
- TF translation into protein at rate $\pi_T$ per TF mRNA
- decay of the TF mRNA at rate $\mu_T$
- decay of the TF protein at rate $\mu_Q$
- miRNA production at rate $\alpha_S(q) = \alpha_S \frac{q}{q+K_S}$
- miRNA decay at rate $\mu_S$
- target mRNA transcription at rate $\alpha_R(Q) = \alpha_R \frac{q}{q+K_R}$
- target mRNA decay at rate $\mu_R$
- target protein production at rate $\pi_R$ per mRNA
- target protein decay at rate $\mu_R$
- free target mRNA binds the miRNA at rate $\beta$ to make a mRNA-miRNA complex
- mRNA-miRNA complex falls apart at rate $\mu_C$, with the mRNA being lost and the miRNA recycled. 

#### Better Description
- TF transcription and decay: $\emptyset \xrightleftharpoons[\mu_T]{\alpha_T} TF_{mRNA}$
- TF translation and decay: $TF_{mRNA} \xrightarrow{\pi_T} TF_{protein} \xrightarrow{\mu_Q} \emptyset$
- miRNA production at rate $\alpha_S(q) = \alpha_S \frac{q}{q+K_S}$
- miRNA production and decay: $\emptyset \xrightleftharpoons[\mu_S]{\alpha_S(q)} miRNA$
- target mRNA transcription at rate $\alpha_R(Q) = \alpha_R \frac{q}{q+K_R}$
- Target mRNA transcription and decay: $\emptyset \xrightleftharpoons[\mu_R]{\alpha_R(q)} mRNA$
- Target protein translation and decay: $mRNA \xrightarrow{\pi_R} Protein \xrightarrow{\mu_R} \emptyset$
- miRNA-mediated mRNA degradation: 
  
  $$miRNA + mRNA \xrightleftharpoons[\mu_C]{\beta} RNA_{complex}\xrightarrow{} miRNA$$

<img src="Images/miRNA-interference_scheme.png" alt="miRNA-interference_scheme" width="550"/>

### Method
#### Gillespie Algorithm
Let us illustrate the algorithm on a very simple system of a reversible dimerization reaction involving two monomers, **A** and **B**. The reactions then are:

$$ {\bf A} + {\bf B} \xrightarrow{r_f} {\bf AB}$$

$$ {\bf AB} \xrightarrow{r_b} {\bf A} + {\bf B}$$
If we denote the number of molecules of each type by $N_A, N_B, N_{AB}$, then the rate of the forward (dimerization) reaction is $r_f N_A N_B$, the rate of the backward (dissociation) reaction is $r_b N_{AB}$ and the rate of any reaction occurring is $R = r_f N_A N_B + r_b N_{AB}$. We then assume that reactions follow a Poisson process, the waiting time to the next reaction being exponentially distributed with rate $R$: $P(t) = R e^{-Rt}$. The mean waiting time would be $\frac{1}{R}$. Note that as reactions take place and the numbers of molecules fluctuate, $R$ also varies in time.

Assuming a reaction did occur, the probability of it being the dimerization is given by 
- $P({\bf A} + {\bf B} \rightarrow {\bf AB}) = \frac{r_f N_A N_B}{R}$ 

while the probability of it being the dissociation is given by 
- $P({\bf AB} \rightarrow {\bf A} + {\bf B}) = \frac{r_b N_{AB}}{R}$ 

So, to simulate this system we can repeat the following steps:
1. Given current numbers of molecules calculate the rate of individual reactions and the total rate $R$
2. Determine the time to the next reaction - sample from the exponential distribution with rate $R$
3. Determine which reaction takes place - sample reaction in proportion to its relative rate
4. Update molecular species involved in the sampled reaction and update the time
   
We have a few more details to sort out, namely how we pick the time and the reaction that takes place at a given step. First, picking which reaction should take place: it is quite intuitive to see that if we view the rates of the reactions as segments of corresponding lengths, line up all the segments next to each other into an interval, and then, when we want to sample reactions, we blindly pick points in the interval (*using the numpy.random.random()* method, which gives us values distributed uniformly in the [0,1] interval), the fraction of times we will land in the segment corresponding to a particular reaction is given by the relative length of that segment in the interval (see figure below).

## Lectures
- Lecture 1: Gillespie Project explanation
- Lecture 2: Python basics
- Lecture 3: Algorithm design
- Lecture 4: Task design
- Lecture 5: Version control
- Lecture 6: Documentation
- Lecture 7: Encapsulation & packaging
- Lecture 8: Repo, executable & docs
- Lecutre 9: Linting, testing & continuous integration
- Lecture 10: Dependency management & contrainerization
- Lecture 11: Final
- Lecture 12: Workflows / Nextflow
- Lecture 13: Wrap up