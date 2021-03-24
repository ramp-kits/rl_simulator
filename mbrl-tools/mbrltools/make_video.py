import os
import json

import click

import gym

import pandas as pd

from rampwf.utils.importing import import_module_from_source


def make_video(submission, agent, seed, epoch):
    """Make a video from a trace.

    The trace is read from
    <env>/submissions/<submission>/mbrl_outputs/<agent>/seed_<seed>/
    epoch_<epoch>/trace.csv

    and the generated video saved in
    <env>/submissions/<submission>/mbrl_outputs/<agent>/seed_<seed>/
    epoch_<epoch>/video/

    Parameters
    ----------
    submission : string
        Submission name.
    agent : string
        Name of the agent.
    seed : int
        The seed used to generate the trace.
    epoch : int
        The epoch of the trace used to generate the video.
    """

    env_module = import_module_from_source('env.py', 'env')

    epoch_path = os.path.join(
        'submissions', submission, 'mbrl_outputs',
        agent, f'seed_{seed}', f'epoch_{epoch}')
    data_path = os.path.join(epoch_path, 'trace.csv')

    video_path = os.path.join(epoch_path, 'video')

    env = gym.wrappers.Monitor(
        env_module.Env(), video_path,
        video_callable=lambda episode_id: True,
        force=True)
    env.reset()

    data = pd.read_csv(data_path)

    metadata_path = os.path.join("data", "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    observation_names = metadata["observation"]
    action_names = metadata["action"]

    X = data.dropna(axis=0)
    for i in range(len(X) - 1):
        current_observation = X[observation_names].to_numpy()[i]
        env.set_state_from_observation((0, current_observation))
        current_action = X[action_names].to_numpy()[i]
        env.step(int(current_action))

    env.close()


@click.command()
@click.option("--submission", default="real_system", show_default=True,
              type=click.STRING,
              help="Model submission. Choose 'real_system' if you want to "
              "use the real environment.")
@click.option('--agent', default='random_shooting', show_default=True,
              type=click.STRING, help="Agent.")
@click.option("--seed", default=0, show_default=True,
              help="The seed used to generate the trace.")
@click.option("--epoch", default=0, show_default=True,
              help="The epoch of the trace used to generate the video.")
def make_video_command(submission, agent, seed, epoch):
    return make_video(submission, agent, seed, epoch)


if __name__ == '__main__':
    make_video_command()
