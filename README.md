# Model-based reinforcement learning

This repository contains a benchmark of model-based reinforcement learning solutions made of probabilistic models and planning agents. This code was used to run the experiments of the paper ["Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?", Balázs Kégl, Gabriel Hurtado, Albert Thomas, ICLR 2021](https://openreview.net/forum?id=p5uylG94S68).

The different systems of the benchmark are located in the `benchmark/` folder. Each system has its own folder where one can
- train and evaluate the different (probabilistic) models on static datasets with common regression metrics (likelihood, R2, ...) by using the `ramp-test` command from [ramp-workflow](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/index.html) (see the [ramp-test command documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/command_line.html#ramp-test) for more information on how to use this command)
- evaluate the models coupled with planning agents in a classical model-based reinforcement learning setup by using the `model-based-rl` command implemented in the `mbrl-tools` package provided in this repository.


## Installation
You can easily install all the required packages with [conda](https://docs.conda.io/projects/conda/en/latest/index.html) and the following procedure:
1. Create a new `conda` environment from `environment.yml` using `conda >= 4.9.2`:
```
conda env create -f environment.yml
```
By default this will create an environment named `mbrl`. You can specify the name of your choice by adding `-n <environment_name>` to the `conda env create` command.

2. Activate the environment with `conda activate mbrl`.

3. Install the [generative regression branch of `ramp-workflow`](https://github.com/paris-saclay-cds/ramp-workflow/pull/193) by running
```
pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git@generative_regression_clean
```

4. Install the `mbrl-tools` package by running `pip install .` in the `mbrl-tools/` directory.

With this installation you can run all the models of the ICLR 2021 paper. If you do not want to run all the models you might only need a subset of the packages listed in `environment.yml`.

Finally, if you want to run the inverted pendulum experiments you need [MuJoCo 2.0](http://www.mujoco.org/index.html) and [mujoco-py](https://github.com/openai/mujoco-py). `mujoco-py` can be installed easily with `pip install mujoco-py`.

## Get started

We will go through the different functionalities using the acrobot system located in `benchmark/acrobot/`. The main structure of this folder follows the one used by [ramp-workflow](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/workflow.html) with a few additional components for the dynamic evaluation (model-based reinforcement learning loop):
* the ramp-workflow `problem.py` file specifying the problem and the training and evaluation protocol.
* the `env.py` file for the Open AI gym environment of the acrobot system
* the `reward_function.py` file for the reward function of the reinforcement learning task
* the `generate_static_trace.py` file, used to generate the static datasets from the real system
* the `data/` folder containing the static datasets generated on the real system 
* the `submissions/` folder containing the different models
* the `agents/` folder containing the different agents

### Static evaluation
To train and evaluate a model located in `submissions/` on a static dataset run `ramp-test --submission <submission_name> --data-label <dataset_name>`. For instance to run the linear model on the dataset generated with a random policy:
```
ramp-test --submission arlin_sigma --data-label random
```
For more information on the `ramp-test` options and generated outputs please refer to the [ramp-workflow documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/workflow.html).

### Dynamic evaluation
To evaluate a model, coupled with a random shooting agent, in a model-based reinforcement learning setup use the `model-based-rl` command. For instance to evaluate the linear model you can run
```
model-based-rl --agent-name random_shooting --submission arlin_sigma
```
You can also choose the number of epochs, the minimum number of steps per epoch or even use an initial trace instead of running a random agent for the first epoch. More information on the different arguments can be obtained by running `model-based-rl --help`.

## Citation
If you use this code please cite our ICLR 2021 paper:
```
@inproceedings{Kegl2021,
  title={Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?},
  author={Kégl, Balázs and Hurtado, Gabriel and Thomas, Albert},
  booktitle={9th International Conference on Learning Representations, {ICLR} 2021},
  year={2021}
}
```