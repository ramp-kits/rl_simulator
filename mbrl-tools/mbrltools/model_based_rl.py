import os
import copy
import click

import numpy as np
import pandas as pd
import torch

from rampwf.utils.importing import import_module_from_source
from rampwf.utils import pickle_trained_model

from .data_processing import get_metadata_dictionary
from .data_processing import rollout

from stable_baselines3.common.vec_env import VecMonitor


def mbrl_run(agent_name, submission,
             n_epochs, min_epoch_steps, min_random_steps,
             episodic_update, initial_trace, model_env_module, num_envs=1,
             seed=99999, partial_fit=False, save_model=True, save_agent=True,
             problem_name=None):
    """Main script of model based RL loop.

    The problem_name argument is used for the purpose of testing.
    """
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    print(f'Random seed: {seed}')

    if min_random_steps is None:
        min_random_steps = min_epoch_steps

    print((f'Using {submission} as model and {agent_name} as '
           f'agent for {n_epochs} epochs with at least {min_epoch_steps} '
           f'steps per epoch and {min_random_steps} random steps.'))

    if problem_name is None:
        problem_module_path = 'problem.py'
    else:
        problem_module_path = problem_name + '.py'
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
        eval_model_env = None
        planning_env = None
    else:
        if model_env_module == 'model_env':
            from .model_env import make_model_env_class
        elif model_env_module == 'numpy_model_env':
            from .numpy_model_env import make_model_env_class
        elif model_env_module == 'sb3_model_vec_env':
            from .sb3_model_vec_env import make_model_env_class
        else:
            raise ValueError('The passed model_env is not supported.')
        submission_path = os.path.join('submissions', submission)
        ModelEnv = make_model_env_class(system_env_object)
        if model_env_module == 'sb3_model_vec_env':
            model_env = ModelEnv(
                submission_path, problem_module, reward_func,
                metadata, output_dir, partial_fit, save_model, seed=None,
                num_envs=num_envs)
            model_env = VecMonitor(model_env)
            eval_model_env = ModelEnv(
                submission_path, problem_module, reward_func,
                metadata, output_dir, partial_fit, save_model, seed=None,
                num_envs=1)
            eval_model_env = VecMonitor(eval_model_env)
            planning_env = ModelEnv(
                submission_path, problem_module, reward_func,
                metadata, output_dir, partial_fit, save_model, seed=None,
                num_envs=1)
        else:
            model_env = ModelEnv(
                submission_path, problem_module, reward_func,
                metadata, output_dir, partial_fit, save_model, seed=None)
            eval_model_env = ModelEnv(
                submission_path, problem_module, reward_func,
                metadata, output_dir, partial_fit, save_model, seed=None)
            planning_env = ModelEnv(
                submission_path, problem_module, reward_func,
                metadata, output_dir, partial_fit, save_model, seed=None)

    # retrieving feature names
    observation_names = metadata["observation"]
    action_names = metadata["action"]
    restart_name = metadata["restart_name"]
    reward_name = metadata["reward"]

    # states are also saved besides the observations in the trace for ease of
    # replay from the collected traces.
    # get the number of states
    system_env.reset()
    n_states = len(system_env.get_numpy_state())

    trace_header = (
        observation_names + action_names + reward_name +
        [restart_name] + ['epoch_id'] +
        [f'state_{i}' for i in range(n_states)])

    if initial_trace:
        # epoch 0 is the initial trace
        print('Epoch 0: Initial trace.')

        # no need to train if the model environment is the real environment
        if hasattr(model_env, 'train_model'):
            model_env.train_model(0)

        epoch_start = 1
        agent = agent_object(model_env, output_dir=output_dir,
                             seed=None,
                             eval_env=system_env_object(),
                             eval_model_env=eval_model_env,
                             planning_env=planning_env,
                             metadata=metadata,
                             epoch=epoch_start)
    else:
        # random agent
        epoch_start = 0
        agent = agent_object(model_env, output_dir=output_dir,
                             random_action=True, seed=None,
                             eval_env=system_env_object(),
                             eval_model_env=eval_model_env,
                             planning_env=planning_env,
                             metadata=metadata,
                             epoch=epoch_start)

    for epoch in range(epoch_start, n_epochs):
        # use the agent on the real system, collect the trace to update the
        # model and update the agent using the updated model
        epoch_output_dir = os.path.join(output_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_output_dir):
            os.makedirs(epoch_output_dir)
        agent.epoch = epoch

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
            if eval_model_env is not None:
                if model_env_module == 'sb3_model_vec_env':
                    eval_model_env.venv.trained_model = copy.deepcopy(
                        model_env.trained_model)
                    planning_env.venv.trained_model = copy.deepcopy(
                        model_env.trained_model)
                else:
                    eval_model_env.trained_model = copy.deepcopy(
                        model_env.trained_model)
                    planning_env.trained_model = copy.deepcopy(
                        model_env.trained_model)

        if save_agent:
            pickle_trained_model(
                epoch_output_dir, agent,
                trained_model_name='trained_agent.pkl', is_silent=False)


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
              type=click.INT,
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
              "submissions/<submission>/mbrl_outputs/<agent_name>/seed_<seed>/"
              "epoch_0/.")
@click.option("--model-env-module", default='model_env', show_default=True,
              type=click.STRING, help="Which model environment module to use. The "
              " default is to use the model_env module based on pandas. For faster "
              " computations use the numpy_model_env one.")
@click.option("--num-envs", default=10, show_default=True, type=click.INT,
              help="The number of environments to consider for the sb3 compatible "
              "vectorized model environment sb3_model_vec_env.")
@click.option("--seed", default=99999, show_default=True,
              help="Seed of the random number generator. Only the numpy and "
              "pytorch global random generators are seeded.")
@click.option("--partial-fit", default=False, show_default=True,
              help="If we want to pass the model from the previous epoch.")
@click.option("--save-model", default=True, show_default=True,
              help="Whether to save the trained_model.pkl at each epoch.")
@click.option("--save-agent", default=True, show_default=True,
              help="Whether to save the trained agent at each epoch.")
def mbrl_run_command(agent_name, submission,
                     n_epochs, min_epoch_steps, min_random_steps,
                     episodic_update, initial_trace, model_env_module, num_envs,
                     seed, partial_fit, save_model, save_agent):
    return mbrl_run(
        agent_name, submission, n_epochs, min_epoch_steps, min_random_steps,
        episodic_update, initial_trace, model_env_module, num_envs, seed, partial_fit,
        save_model, save_agent,
    )


if __name__ == "__main__":
    mbrl_run_command()
