# Equilibria in an Economy with Heterogeneous Agents

## Basis
This repo contains an implementation of Foundation, a framework for flexible, modular, and composable environments that **model socio-economic behaviors and dynamics in a society with both agents and governments**.

Foundation provides a [Gym](https://gym.openai.com/)-style API:

- `reset`: resets the environment's state and returns the observation.
- `step`: advances the environment by one timestep, and returns the tuple *(observation, reward, done, info)*.

The Foundation of this repository is from:

```
@misc{2004.13332,
 Author = {Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C. Parkes, Richard Socher},
 Title = {The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies},
 Year = {2020},
 Eprint = {arXiv:2004.13332},
}
```

## Heterogeneous Agents
The agents were adjusted to have equality and the state of nature as part of their reward structure. For more information, we refer to our report.

## Methods
The original training procedure of the AI Economist paper trains the tax agent and active agents concurrently (after the pretraining phase for the active agents).

In order to find Stackelberg Equilibria, one has to train the agents after each other.
Following the procedure of [Gerstgrasser and Parkes](https://proceedings.mlr.press/v202/gerstgrasser23a/gerstgrasser23a.pdf), we implemented meta learning algorithms where the followers (active agents) are trained first to behave well given information about the leaders policy(tax agent), and the leader is trained afterwards.
For that several new capabilities had to be added.

The config files of the experiments can be found [here](https://www.github.com/Thahit/AI_Econ/tree/master/tutorials/rllib/runs). Some experiments include:
- [Assuming that the current tax brackets are enough for the followers to plan optimally](https://www.github.com/Thahit/AI_Econ/tree/master/tutorials/rllib/runs/one_tax_bracket_oracle)
- [Monte carlo estimate for policy of the leader using a rolling window](https://www.github.com/Thahit/AI_Econ/tree/master/tutorials/rllib/runs/MC_window)
- [Tax agent can select taxes from a predetermined set and shift them. The rolling window approach is used again](https://www.github.com/Thahit/AI_Econ/tree/master/tutorials/rllib/runs/select_rules)


## Installation Instructions

To get started, you'll need to have Python 3.7+ installed.

### Using pip

Simply use the Python package manager:

```python
pip install ai-economist
```

### Installing from Source

1. Clone this repository to your local machine:

  ```
   git clone www.github.com/Thahit/AI_Econ
   ```

2. Create a new conda environment (named "ai-economist" below - replace with anything else) and activate it

  ```pyfunctiontypecomment
   conda create --name ai-economist python=3.7 --yes
   conda activate ai-economist
   ```

3. Install as an editable Python package
  ```pyfunctiontypecomment
   cd ai-economist
   pip install -e .
   ```


You can then simply run `aiecon` once to activate the conda environment.

### Testing your Install

To test your installation, try running:

```
conda activate ai-economist
python -c "import ai_economist"
```

## Getting Started

To familiarize yourself with Foundation, check out the tutorials in the `tutorials` folder. You can run these notebooks interactively in your browser on Google Colab.

### Multi-Agent Simulations

- [economic_simulation_basic](https://www.github.com/Thahit/AI_Econ/tree/master/tutorials/economic_simulation_basic.ipynb) ([Try this on Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb)!): Shows how to interact with and visualize the simulation.
- [economic_simulation_advanced](https://www.github.com/salesforce/Thahit/AI_Econ/tree/master/tutorials/economic_simulation_advanced.ipynb) ([Try this on Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/economic_simulation_advanced.ipynb)!): Explains how Foundation is built up using composable and flexible building blocks.
- [optimal_taxation_theory_and_simulation](https://github.com/Thahit/AI_Econ/tree/master/tutorials/optimal_taxation_theory_and_simulation.ipynb) ([Try this on Colab](https://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/optimal_taxation_theory_and_simulation.ipynb)!): Demonstrates how economic simulations can be used to study the problem of optimal taxation.

### Multi-Agent Training
- [multi_agent_gpu_training_with_warp_drive](https://github.com/Thahit/AI_Econ/tree/master/tutorials/multi_agent_gpu_training_with_warp_drive.ipynb) ([Try this on Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/multi_agent_gpu_training_with_warp_drive.ipynb)!): Introduces our multi-agent reinforcement learning framework [WarpDrive](https://arxiv.org/abs/2108.13976), which we then use to train the COVID-19 and economic simulation.
- [multi_agent_training_with_rllib](https://github.com/Thahit/AI_Econ/tree/master/tutorials/multi_agent_training_with_rllib.ipynb) ([Try this on Colab](http://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/multi_agent_training_with_rllib.ipynb)!): Shows how to perform distributed multi-agent reinforcement learning with [RLlib](https://docs.ray.io/en/latest/rllib/index.html).
- [two_level_curriculum_training_with_rllib](https://github.com/Thahit/AI_Econ/tree/master/tutorials/two_level_curriculum_learning_with_rllib.md): Describes how to implement two-level curriculum training with [RLlib](https://docs.ray.io/en/latest/rllib/index.html).

To run these notebooks *locally*, you need [Jupyter](https://jupyter.org). See [https://jupyter.readthedocs.io/en/latest/install.html](https://jupyter.readthedocs.io/en/latest/install.html) for installation instructions and [(https://jupyter-notebook.readthedocs.io/en/stable/](https://jupyter-notebook.readthedocs.io/en/stable/) for examples of how to work with Jupyter.

## Structure of the Code

- The simulation is located in the `ai_economist/foundation` folder.

The code repository is organized into the following components:

| Component | Description |
| --- | --- |
| [base](https://www.github.com/Thahit/AI_Econ/tree/master/ai_economist/foundation/base) | Contains base classes to can be extended to define Agents, Components and Scenarios. |
| [agents](https://www.github.com/Thahit/AI_Econ/tree/master/ai_economist/foundation/agents) | Agents represent economic actors in the environment. Currently, we have mobile Agents (representing workers) and a social planner (representing a government). |
| [entities](https://www.github.com/Thahit/AI_Econ/tree/master/ai_economist/foundation/entities) | Endogenous and exogenous components of the environment. Endogenous entities include labor, while exogenous entity includes landmarks (such as Water and Grass) and collectible Resources (such as Wood and Stone). |
| [components](https://www.github.com/Thahit/AI_Econ/tree/master/ai_economist/foundation/components) | Components are used to add some particular dynamics to an environment. They also add action spaces that define how Agents can interact with the environment via the Component. |
| [scenarios](https://www.github.com/Thahit/AI_Econ/tree/master/ai_economist/foundation/scenarios) | Scenarios compose Components to define the dynamics of the world. It also computes rewards and exposes states for visualization. |


## Simulation Cards: Ethics Review and Intended Use

Please see our [Simulation Card](https://github.com/Thahit/AI_Econ/tree/master/Simulation_Card_Foundation_Economic_Simulation_Framework.pdf) for a review of the intended use and ethical review of our framework.

## License

Foundation and the AI Economist are released under the [BSD-3 License](LICENSE.txt).

