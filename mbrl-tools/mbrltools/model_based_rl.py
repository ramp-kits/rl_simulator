import os

import click

import numpy as np
import pandas as pd
import torch

from rampwf.utils.importing import import_module_from_source

from .data_processing import get_metadata_dictionary
from .data_processing import rollout
from .model_env import make_model_env_class


@click.command()
@click.option('--agent-name', default='random_shooting', show_default=True,
              type=click.STRING, help="Agent.")
@click.option("--submission", default="real_system", show_default=True,
              type=click.STRING,
              help="Model submission. Choose 'real_system' if you want to "
              "use the real environment.")
@click.option("--n-epochs", default=100, show_default=True, type=click.INT,
              help="The number of epochs. If the submission is not the real "
              "system, the model is updated at each epoch. If initial-trace "
              "is set to True the first epoch is assumed to be the initial "
              "trace.")
@click.option("--min-epoch-steps", default=200, show_default=True,
              type=click.INT,
              help="The minimum number of steps for each epoch given that "
              "each epoch ends by a complete episode.")
@click.option("--min-random-steps", default=None, show_default=True,
              help="The minimum number of steps done at the first epoch"
              " with the random policy if initial-trace is set to False. "
              "If None then it is equal to min-epoch-steps.")
@click.option("--episodic-update", default=False, show_default=True,
              type=click.BOOL,
              help="Whether to update the model after each episode such that "
              "one epoch is exaclty one episode.")
@click.option("--initial-trace", default=False, show_default=True,
              type=click.BOOL, help="Whether an initial trace is available. "
              "If True, the initial trace should be stored under trace.csv in "
              "output/<submission>/<agent_name>/seed_<seed>/epoch_0/.")
@click.option("--seed", default=0, show_default=True,
              help="Seed of the random number generator. Only the numpy and "
              "pytorch global random generators are seeded.")
def model_based_rl(agent_name, submission,
                   n_epochs, min_epoch_steps, min_random_steps,
                   episodic_update, initial_trace,
                   seed):
    """Main script of model based RL loop."""

    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    print(f'Random seed: {seed}')

    if min_random_steps is None:
        min_random_steps = min_epoch_steps

    print((f'Using {submission} as model and {agent_name} as '
           f'agent for {n_epochs} epochs with at least {min_epoch_steps} '
           f'steps per epoch and {min_random_steps} random steps.'))

    problem_module_path = 'problem.py'
    problem_module = import_module_from_source(problem_module_path, 'problem')

    env_module_path = 'env.py'
    env_module = import_module_from_source(env_module_path, 'env')
    system_env_object = env_module.Env
    system_env = system_env_object()
    system_env.seed()

    # reward function
    reward_module_path = 'reward_function.py'
    reward_module = import_module_from_source(
        reward_module_path, 'reward_function')
    reward_func = reward_module.reward_func

    # agent
    agent_module_path = os.path.join('agents', agent_name + '.py')
    agent_module = import_module_from_source(agent_module_path, agent_name)
    agent_object = agent_module.Agent

    # metadata
    metadata_path = os.path.join('data', 'metadata.json')
    metadata = get_metadata_dictionary(metadata_path)

    # create a directory to store the results
    output_dir = os.path.join(
        'submissions', submission, 'mbrl_outputs', agent_name, f'seed_{seed}')

    # model
    if submission == 'real_system':
        model_env = system_env
    else:
        submission_path = os.path.join('submissions', submission)
        ModelEnv = make_model_env_class(system_env_object)
        model_env = ModelEnv(
            submission_path, problem_module, reward_func,
            metadata, output_dir, seed)

    # retrieving feature names
    observation_names = metadata["observation"]
    action_names = metadata["action"]
    restart_name = metadata["restart_name"]
    reward_name = metadata["reward"]

    trace_header = (
        observation_names + action_names + reward_name +
        [restart_name] + ['epoch_id'])

    epoch_output_dir = os.path.join(output_dir, f'epoch_0')
    if initial_trace:
        # epoch 0 is the initial trace
        print('Epoch 0: Initial trace.')

        # no need to train if the model environment is the real environment
        if hasattr(model_env, 'train_model'):
            model_env.train_model()

        agent = agent_object(model_env, epoch_output_dir=epoch_output_dir,
                             seed=None)
        epoch_start = 1
    else:
        # random agent
        agent = agent_object(model_env, epoch_output_dir=epoch_output_dir,
                             random_action=True, seed=None)
        epoch_start = 0

    for epoch in range(epoch_start, n_epochs):
        # use the agent on the real system, collect the trace to update the
        # model and update the agent using the updated model

        if epoch == 0:
            # random policy
            min_rollout_steps = min_random_steps
        else:
            min_rollout_steps = min_epoch_steps
            agent.random_action = False

        trace = rollout(
            system_env, len(action_names),
            epoch=epoch, min_epoch_steps=min_rollout_steps,
            agent=agent, episodic_update=episodic_update)

        # save new trace to disk
        epoch_output_dir = os.path.join(output_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_output_dir):
            os.makedirs(epoch_output_dir)
        trace_path = os.path.join(epoch_output_dir, 'trace.csv')
        trace_df = pd.DataFrame(data=trace, columns=trace_header)
        trace_df.to_csv(trace_path, index=False)

        # update model if it remains epochs to compute.
        if epoch <= n_epochs - 2 and hasattr(model_env, 'train_model'):
            model_env.train_model(epoch=epoch)


if __name__ == "__main__":
    model_based_rl()
