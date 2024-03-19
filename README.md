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
