# Model-based reinforcement learning

This repository contains a benchmark of model-based reinforcement learning solutions made of probabilistic models and planning agents. This benchmark was used to run the experiments of the paper ["Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?", Balázs Kégl, Gabriel Hurtado, Albert Thomas, ICLR 2021](https://openreview.net/forum?id=p5uylG94S68). You can also check the [associated blog post](https://towardsdatascience.com/model-based-micro-data-reinforcement-learning-cabe95990664) for the general context and a summary of this paper.

The different systems of the benchmark are located in the `benchmark/` folder. Each system has its own folder where one can
- train and evaluate the different (probabilistic) models on static datasets with common regression metrics (likelihood, R2, ...) by using the `ramp-test` command from [ramp-workflow](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/index.html) (see the [ramp-test command documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/command_line.html#ramp-test) for more information on how to use this command),
- simulate traces using the learned models on the static data sets using the `mbrl-simulate` command implemented in the `mbrl-tools` package provided in this repository, mainly to compute longer-horizon scores but also for visualization, and
- evaluate the models coupled with planning agents in a classical model-based reinforcement learning setup by using the `mbrl-run` command implemented in the `mbrl-tools` package provided in this repository.


## Installation

You can install all the required packages with [conda](https://docs.conda.io/projects/conda/en/latest/index.html) and the following procedure:
1. Create a new `conda` environment using `conda >= 4.9.2`:
```
conda create -n mbrl python=3.8
```

2. Activate the environment with `conda activate mbrl`.

3. Install `pytorch`:
```
pip install torch==1.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install the packages listed in `requirements.txt`:
```
pip install -r requirements.txt
```

5. Install the [`generative_regression_clean` branch of `ramp-workflow`](https://github.com/paris-saclay-cds/ramp-workflow) by running
```
pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git@180db08499a3cd8b3894fa69a3d6e42d1ee2cafb
```

4. Install the `mbrl-tools` package by running `pip install .` in the `mbrl-tools/` directory.

Finally, if you want to run the inverted pendulum experiments you need [MuJoCo 2.0](http://www.mujoco.org/index.html) and [mujoco-py](https://github.com/openai/mujoco-py). `mujoco-py` can be installed easily with `pip install mujoco-py`.

## Getting started

We will go through the different functionalities using the acrobot system located in `benchmark/acrobot/`. The main structure of this folder is based on the one required by [ramp-workflow](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/workflow.html) with a few additional components for the dynamic evaluation (model-based reinforcement learning loop):
* the ramp-workflow `problem.py` file specifying the problem and the training and evaluation protocol
* the `env.py` file for the Open AI gym environment of the acrobot system
* the `reward_function.py` file for the reward function of the reinforcement learning task
* the `generate_static_trace.py` file used to generate the static datasets from the real system
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

To evaluate a model, coupled with a random shooting agent, in a model-based reinforcement learning setup use the `mbrl-run` command. For instance to evaluate the linear model you can run
```
mbrl-run --submission arlin_sigma --agent-name random_shooting_n100_h10
```
The `--submission` option name was inherited from the terminology used by `ramp-test`. Other options include the number of epochs, the minimum number of steps per epoch, using an initial trace instead of running a random agent for the first epoch. More information on the different options can be obtained by running `model-based-rl --help`.

For the agents that depend on `stable_baselines3`, you need to use the associated model environment:
```
mbrl-run --submission arlin_sigma --agent-name DQN_offline_vec_env --model-env-module sb3_model_vec_env
```

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