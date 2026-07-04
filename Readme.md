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
We consider several algorithms from the literature as baselines for comparison. As a learning-based approach, we adopt the pathwise differentiation method (PDM) (Tzen & Raginsky, 2019; Liu et al., 
2019). PDM can be used for joint optimization over the entire time horizon, which is needed for a time-varying optimization problem. 
In addition, we consider the proximal stochastic gradient (PSG) method from (Cutler et al., 2023) as a gradient-based baseline. This algorithm addresses convex optimization problems under distribution drift and is a stochastic gradient descent methods. PSG tracks the optimal decision process by using a stochastic algorithm with iterate averaging.

### (1) Least-Squares Recovery
We consider a least-squares recovery problem with distribution drift. 
Specifically, this problem aims to recover a variable that follows a non-stationary distribution, based on observations following a Gaussian process with thime-varying mean.  

Figures below compares our SPF algorithm to the learning-based PDM and gradient-based PSG methods, using two performance metrics: (i) the optimality distance, which measures how close the
recovered decision process is to the true target, and (ii) the objective suboptimality, which measures the difference between the objective evaluated at recovered process and the target objective.

|Optimality distance  on the least-squares recovery problem |Objective suboptimality  on the least-squares recovery problem|
|:-------------------------:|:-------------------------:|
|  <img src="images/Recovery1.png" alt="Alt Text" style="width:1000px;"> | <img src="images/Recovery2.png" alt="Alt Text" style="width:1000px;"> | 
|Although the PDM algorithm achieves an optimality distance close to that of SPF, the SPF method outperforms both PDM and PSG | The same trend holds for objective suboptimality, where SPF exhibits the lowest error among the learning-based and gradient-based approaches |

### (2) Logistic Regression
We now consider an ℓ_2-regularized logistic regression problem leveraged in (Cutler et al., 2023) with a random sequence of soft labels.

Figures below illustrate the performance of the SPF algorithm compared with the learning-based PDM and gradient-based PSG methods in terms of the optimality distance and the objective suboptimality.

|Optimality distance  on the  logistic regression problem |Objective suboptimality  on the  logistic regression problem|
|:-------------------------:|:-------------------------:|
|  <img src="images/Regression1.png" alt="Alt Text" style="width:1000px;"> | <img src="images/Regression2.png" alt="Alt Text" style="width:400px;"> | 
|The SPF algorithm achieves a superior optimality distance relative to the PDM approach and performs comparably to the PDM method overall | The same pattern is observed for the objective suboptimality, where SPF demonstrates improved performance relative to the gradient-based PSG and comparable performance to the conventional learning-based PDM method|

## 🛠️ Dependencies
Trained and Tested on:
```
Python 3.11
PyTorch
NumPy
scipy.special
```
Graphs
```
matplotlib
```

## 📚 Reference

This repository accompanies:
Mohsen Amidzadeh, Lauri Viitasaari, Mario Di Francesco, "Non-Parametric Optimization for Scalable Learning in Stochastic Decision Problems", ICML, 2026.🔗 https://openreview.net/forum?id=rD79qEJ5iR
```
@article{
amidzadeh2026NonparametricStochasticOptimization,
title={Non-Parametric Optimization for Scalable Learning in Stochastic Decision Problems},
author={Mohsen Amidzadeh, Lauri Viitasaari, Mario Di Francesco},
journal={International  Conference on Machine Learning (ICML)},
year={2026},
url={https://openreview.net/forum?id=rD79qEJ5iR},
}
```
