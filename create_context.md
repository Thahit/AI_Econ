# apply a LSTM to provide leader responds

## Question

how we can fully characterize the leader strategy through queries

how the effect of the leader policy on the follower can be fully understood with a minimal amount of queries

## Idea: multitask and meta-RL literature to infer context as a sequence of queries

Below are the listed papers

## Efficient Off-Policy Meta-RL

### Definitions

* distribution of tasks p(T)
  * different tax brackets as a distribution over multiple planner polices

* the meta-training process learns a policy that adapts to the task at hand by conditioning on the history of past transitions, which we refer to as context c
  * we can query actions of the planner given the observation of the env

* Condition the policy as \pi_θ (a|s, z) in order to adapt its behavior to the task
  * create embeddings of the planner networks to train a latent contect representing all possible policies

* We train an inference network qφ(z|c), parameterized by φ, that estimates the posterior p(z|c)
* optimizing qφ(z|c) to reconstruct the MDP by learning a predictive models of reward and dynamics
* qφ(z|c) can be optimized in a model-free manner to model the state-action value functions

* modeling qφ(z|c) as a product of independent factors
* use Gaussian factors Ψφ(z|cn) = N (f μφ (cn), f σφ (cn)), which result in a Gaussian posterior.