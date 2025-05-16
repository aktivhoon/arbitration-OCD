# Project
Neurocomputational model of impaired arbitration between model-based and model-free learning in OCD (Kim, T. et al. In press, *Brain*)

> refer to [Lee, S.W. et al. 2014, *Neuron*](https://doi.org/10.1016/j.neuron.2013.11.028):
Computations underlying arbitration between MB and MF learning

#### Minor modifications from the original version
1. debugging the backward update algorithm
2. reliability -> uncertainty
3. tuning the boundaries of model parameters


---
# Scripts
## simul_Arb
- run simulation of model fitting using the codes below
- includes pretraining and main training sections
- interation for each trial/stage

## optim_Arb
- run model fitting using a maximum likelihood estimation method
- return fitted model parameters
- multiple seeds testing recommend for optimization
- for model validation, necessary to run parameter recovery along with action generation using the optimized model

## batch_pcs
- prepared for parallel computing
- load behavioral data for model fitting

---
# Codes
## Main computations
### Model_RL2
- training the MB (fwd + bwd)/MF (sarsa) RL agents.
- imitation learning (**decision_behavior_data_save**) for model fitting.

### Bayesian_Arb
- estimation of 1) the prediction uncertainty of each learning (**m?_inv_Fano**) and 2) the dynamic weight between the two strategies (**m1_prob**).
- finally, integrating the action values for the arbitration system using the weight variable.

## Environment and data structure
### Model_Map_Init2
- set the two-step decsion task structure (**myMap**).

### Model_RL_Init
- contrsuct the data structures for state (**state_history**), action (**action_history**),
learning information (**SPE_history**, **T**, etc.).

### Bayesian_Arb_Init
- construct the data structures for uncertainty-based arbitration (**Bayesian_Arb**)

### StateSpace_v1
- transitioning subjects/agents to a next state according to state-transition probabilities.
- **opt.use_data=1** for model fitting.

### StateClear
- Clear environment and action data after a trial (S1->S2->S3).





## 🔧 Bug Fixes & Important Code Corrections

> These fixes were made to ensure the correctness and stability of the arbitration and RL computations.

### 1. 🐞 SPE Index Misalignment
- **File**: `Bayesian_Arb.m`
- **Line**: ~30  
- **Issue**: The first index of `SPE_history` should be neglected (always 0), so `index + 1` must be used.  
- **Fix**: Replaced `SPE_history(index)` with `SPE_history(index + 1)` to skip the initial dummy value and correctly align subsequent SPEs.

---

### 2. 🐞 Incorrect Use of Reliability Instead of Uncertainty
- **File**: `Bayesian_Arb.m`
- **Lines**: ~45–60  
- **Issue**: Original code computed arbitration weights using **reliability** (e.g., `1 - omega/RPE_max`).  
- **Fix**: Switched to using **uncertainty** directly and adjusted transition weighting as `1 - uncertainty`.

---

### 3. 🐞 Overestimated Q-values Due to Reward Duplication
- **File**: `simul_Arb.m`
- **Lines**: Reward assignment logic near trial iteration  
- **Issue**: Reward from the final state of one episode was carried into the initial state of the next, especially when `state == S1`.  
- **Effect**: Q-values exceeded maximum expected reward (e.g., 48, 50 > max 40).  
- **Fix**: Added a condition to nullify the reward at the initial state of a new episode.

---

### 4. 🐞 Double Backward Update in 2-Step Task
- **File**: `simul_Arb.m`
- **Lines**: Backward update block in loop over steps  
- **Issue**: Backward update was applied twice (step 1 and step 2).  
- **Fix**: Applied backward update **only once** per trial when reward context changes.

---

### 5. 🐞 Arbitrary Transition Threshold for Habitual Shift
- **File**: `batch_pcs.m`
- **Line**: Parameter setup (~line 15–20)  
- **Note**: For proper MB→MF arbitration dynamics, threshold was set to `0.1` instead of default. This setting enhances the likelihood of observing habitual patterns during fitting.
