# Stochastic Path Follower (SPF): Non-Parametric Optimization for Scalable Learning in Stochastic Decision Problems

## 📑 introduction

**SPF** (Stochastic Path Follower ) is a PyTorch implementation for solving (time-varying) stochastic optimization problems where distribution drift occurs or the environmnet is non-stationary.

🎯 **SPF** addresses time-varying stochastic optimization (TV-SO) problems as the stochastic couterpart of conventional time-varying optimization challenges.
 It finds non-parametric optimality conditions via Malliavin calculus. Its non-parametric framework provides a learning mechanism insensitive to the parameterization dimension, the challenge which most of the baselines such as adjoint sensitivity models and path-wise differentiation methods (PDMs) struggle with. <br>
🎯 **SPF** provides a scalable neural algorithm for solving stochastic optimization challenges under distribution drift, and as such it is parallel to PDMs and adjoint sensitivity approaches, but far less sensitive to the parameterization dimension. <br>
🎯 **SPF** is competitive with the baseline algorithms from both complexity and performance perspectives.

## ⚙️ Overview of the SPF algotithm:
SPF consists of three main phases as follows.

**(i) Forward pass.** Simulate a neural SDE, with drift and diffusion functions, to obtain the stochastic paths of decision process. In addition, compute the Malliavin derivatives of the decisoin process.

**(ii)  Loss evaluation.** Compute an energy-functional loss associated with the considered TV-SO problem.

**(iii)  Parameter update.** Update the parameters of the neural drift and diffusion functions  using an Adam-type optimizer.

A pseudo-code of the SPF algorithm is provided below.
|  Diagram of SPF algorithm |
| :-------------------------:| 
|  <img src="images/SPF.png" alt="Alt Text" style="width:400px;">  |  
|  |


## 🧪 Usage
- To train the RL agent on a FB-MDP: run `train.py`
- To test a preTrained network : run `test.py`

  ### Experiments:
  Notice that current (standard) RL baselines/experiments are not applicable to forward-backward MDPs. Hence, we go beyond them to assess our algorithm on real-world problems characterized by FB-MDPs.
  For this, we have provided two large-scale experiments falling within FB-MDP frameworks.
  The first experiment is a edge-cashing problem in the context of communication networkings,
  and the second experiment is a computation offloading problem in the domain of cloud computing systems. 
  These experiments are provided in the folder [*experiment*]. Moreover, there is a [Readme] in that folder that explains these experiments and thier hyper-parameters.

  For the edge caching experiment, please uncomment the syntax **from environments.EdgeCaching import NetEnv** and for the computation odffloading experiment uncomment the syntax **from environments.ComputationOffloading import NetEnv**
  in the train.py or test.py.

### 🔧 Algorithm hyperparameters:
-    **Print_freq**        :                        The frequency based on which the training results should be printed. (after how many episodes).
 -   **Save_model_freq**    :                       The frequencyt based on which  the parameters of model should be saved.
  -  **AverageFrequency**   :                       The frequency based on which  the cumulative rewards should be averagd for the printing and logging purposes.
  -  **N_MCS**             :                        Number of Monte-Carlo Samples for the *episodic MCS-average* add-on.
   - **EpisodeNumber**   :                          Number of training episodes.
  -  **TimeSlots**         :                        Number of time-steps in each episode.
   - **LearningRate**    :                          Learning-Rate of the FB-MOAC algorithm, for *multi-objective actor* and *forward/backward critics*.
   - **SmoothingFactor** :                          The smoothing factor of the episodic MCS-average  add-on.
  -  **DiscountFactor**   :                         Discount-factor related to the cumulative rewards.
  -  **PreferenceCoeff**                            Preference parameter, for forward and backward rewards, to extract a Pareto-front. (e.g. for a problem with 2 forward rewards and one backward one, one may set PreferenceCoeff = torch.tensor([p1, p2, p3]) which means that forward rewards have p1 and p2 preferences and backward reward has p3. )

##### Note :
  - For each environment, the hyper-parameters need fine-tuning. FB-MOAC can also be used for forward-only multi-objective MDP problems.


## 📈  Results
Foe the comparison purposes, we consider two baseline algorithms, namely **PPO** and a **multi-objective actor-critic (MOAC)**. However, to be able to use these algorithms, which are not designed for forward-backward MDPs, we modified the respective problems by replacing the backward reward(s) with a related one(s) (for fairness) so that the backward MDP can be safely removed. Figures below show both the performance of FB-MDP and its comparison with  **PPO** (called F-PPO) and the **multi-objective actor-critic** (called F-MOAC).

### (1) Edge-Cahing  Experiment.
please refer to the Readme file in the environment folder to see a brief explanation for this experiment. 
Full details are given in the paper. 

|Obtained Pareto-set of FB-MOAC for edge-caching experiment |
|:-------------------------:|
|  <img src="images/Results/multiobjective_comparison_preferences.png" alt="Alt Text" style="width:1000px;"> | 


| Training Comparison of FB-MOAC against PPO and MOAC (a multi-objective A2C) | Comparison of Final Solutions of FB-MOAC against PPO and MOAC (a multi-objective A2C) | 
| :-------------------------:|:-------------------------:|
|  <img src="images/Results/FBMOAC_FA2C_PPO.png" alt="Alt Text" style="width:400px;"> |  <img src="images/Results/test-multicast1.png" alt="Alt Text" style="width:400px;"> | 



### (2) Computation-Offloading  Experiment
please refer to the Readme file inthe environment folder to see a brief explanation for this experiment. 
Full details are given in the paper. 


| Performance of solution of FB-MOAC for computation offloading experiment  | Histogram of solution of FB-MOAC for computation offloading experiment  |
| :-------------------------:|:-------------------------:|
|  <img src="images/Results/performance-offload-fb-moac.png" alt="Alt Text" style="width:400px;"> | <img src="images/Results/offload-fb-moac.png" alt="Alt Text" style="width:400px;"> |


| Comparison of solution of F-PPO for computation offloading experiment  | Comparison of solution of F-MOAC for computation offloading experiment  |
| :-------------------------:|:-------------------------:|
|  <img src="images/Results/offload-f-ppo.png" alt="Alt Text" style="width:400px;"> |  <img src="images/Results/offload-f-moac.png" alt="Alt Text" style="width:400px;"> | 



## 🛠️ Dependencies
Trained and Tested on:
```
Python 3.11
PyTorch
NumPy
scipy.special
```
Training Environments 
```
Edge-caching
Computation-offloading
```
Graphs
```
matplotlib
```

## 📚 Reference

This repository accompanies:
Mohsen Amidzade, Mario Di Francesco, "FB‑MOAC: Forward–Backward Multi‑Objective Actor‑Critic", TMLR, 2025.🔗 https://openreview.net/forum?id=li5DyC6rfS
```
@article{
amidzadeh2025fbmoac,
title={{FB}-{MOAC}: A Reinforcement Learning Algorithm for Forward-Backward Markov Decision Processes},
author={Mohsen Amidzadeh and Mario Di Francesco},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=li5DyC6rfS},
}
```
