import os
import json

import pathlib
import collections

import pandas as pd
import numpy as np

from tqdm import tqdm
from natsort import natsorted

from torch.utils.tensorboard import SummaryWriter

Experience = collections.namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'new_state'])


def get_metadata_dictionary(metadata):
    """Get metadata dictionary.

    metadata : string, path or dict
        If string or path, path of the metadata file containing the metadata.
        Should be a json file.
        If dict, dictionary containing the metadata. In this case nothing has
        to be done.

    Returns
    -------
    metadata : dict
        Dictionary containing the metadata.
    """

    if isinstance(metadata, str) or isinstance(metadata, pathlib.Path):
        with open(metadata, "r") as json_file:
            metadata = json.load(json_file)
    elif not isinstance(metadata, dict):
        raise ValueError('metadata should either be a string or a dictionary')

    return metadata


def read_data_with_metadata(trace_path, metadata):
    """Read a trace file with the encoding information given by its metadata.
    """
    # find the encoding of the csv
    try:
        enc = metadata["encoding"]
    except KeyError:
        enc = None

    return pd.read_csv(trace_path, encoding=enc, engine='pyarrow')


def preprocess_time(data, metadata):
    """Preprocess time.

    Creates a timestamp if absent in metadata and sorts values by time.
    """
    timestamp_name = metadata.get('timestamp_name')
    if timestamp_name is None:
        return data

    if timestamp_name == "":
        timestamp_name = "fake_ts"
        data[timestamp_name] = data.index

    data[timestamp_name] = pd.to_datetime(data[timestamp_name])
    data.sort_values(by=[timestamp_name], inplace=True)
    data.set_index([timestamp_name], inplace=True)

    return data


def get_first_episodes(trace_path, metadata, n_episodes=None):
    """Return the first episodes of a trace.

    Parameters
    ----------
    trace_path : string
        Path of the trace to load and reformat.
    metadata : dict
        Dictionary containing the metadata of the trace.
    n_episodes : int
        The number of episodes of the input trace to return. If None the whole
        trace is returned.

    Returns
    -------
    trace_df : pandas DataFrame
        Dataframe containing the first episodes.
    """
    metadata = get_metadata_dictionary(metadata)
    restart_name = metadata["restart_name"]
    trace_df = read_data_with_metadata(trace_path, metadata)

    if n_episodes is not None:
        episode_start_ind = np.where(
            (trace_df[restart_name] == 1).to_numpy())[0]
        if n_episodes > len(episode_start_ind):
            raise ValueError(
                'The number of episodes to get from the initial trace is '
                'larger than the total number of episodes')
        if n_episodes < len(episode_start_ind):
            # if it is equal then we have nothing to do this corresponds to
            # the whole initial trace
            first_excluded_ind = episode_start_ind[n_episodes]
            trace_df = trace_df.iloc[:first_excluded_ind]

    return trace_df


def rollout(system_env, n_action_features,
            epoch=0, min_epoch_steps=1000, n_episodes=None,
            agent=None, episodic_update=False,
            n_cum_steps=0,
            tensorboard_path=None):
    """Perform a rollout on the environment using the agent.

    Parameters
    ----------
    system_env : object
        System environment. Needs a reset and step methods.
        If agent is None, the environment should have an action_space
        attribute with a sample method.
    n_action_features : int
        Number of action features.
    epoch : int
        Epoch index.
    min_epoch_steps : int
        Minimum number of steps of the rollout. The rollout terminates on
        a complete episode
    n_episodes : int
        Number of episodes to run. Takes the priority over min_epoch_steps if
        set.
    agent : object
        Agent. If not None, needs an act method.
        If None then system_env.action_space.sample is used.
    episodic_update : bool
        Whether to update the model after each episode, in which case one epoch
        is exaclty one episode.
    n_cum_steps : int
        Number of cumulative steps. Used for tensorboard logging.
    tensorboard_path : string
        Path to the tensorboard directory.

    Return
    ------
    trace : list of numpy arrays
        Trace collected from the rollout.
    """
    print('\nEpoch', epoch)

    if n_episodes is not None:
        min_epoch_steps = np.inf
    else:
        n_episodes = np.inf

    if tensorboard_path is not None:
        writer_rollout = SummaryWriter(
            tensorboard_path,
            flush_secs=1)

    # epoch trace data
    epoch_step = 0
    trace = []
    n_done_episodes = 0

    while epoch_step < min_epoch_steps and n_done_episodes < n_episodes:
        if hasattr(agent, 'reset'):
            agent.reset()
        observation = system_env.reset()
        state = system_env.get_numpy_state()

        rewards = []
        episode_step = 0
        for episode_step in tqdm(
                range(system_env.max_episode_steps), initial=1,
                total=system_env.max_episode_steps, desc='Steps'):
            # restart = 1 if first episode step, otherwise 0
            restart = int(not(episode_step))
            if agent is None:
                action = system_env.action_space.sample()
            else:
                action = agent.act(observation, restart)
            new_observation, reward, done, _ = system_env.step(action)
            rewards.append(reward)

            if hasattr(agent, 'exp_buffer'):
                # add real data to the experience buffer so that the
                # agent can be trained with these data
                agent.exp_buffer.append(
                    Experience(
                        observation, action, reward, 0, new_observation))

            trace_step = np.hstack(
                (observation, action, reward, restart, epoch, state))
            trace.append(trace_step)

            # update observation
            observation = new_observation
            state = system_env.get_numpy_state()

            if done:
                break

        # save last observation before a reset
        n_nans = n_action_features + 1  # NaNs for actions and reward
        last_obs = np.hstack(
            (observation, np.repeat(np.nan, n_nans), 0, epoch, state))
        trace.append(last_obs)

        episode_step += 1  # this is to take into account the fact that we start at 0
        epoch_step += episode_step
        n_done_episodes += 1

        print('Number of episode steps:', episode_step)
        print('Mean reward:', np.mean(rewards))
        print('Return:', np.sum(rewards))
        print('Number of epoch steps:', epoch_step)

        n_cum_steps += episode_step
        if tensorboard_path is not None:
            writer_rollout.add_scalar(
                'Mean reward', float(np.mean(rewards)), global_step=n_cum_steps)
            writer_rollout.add_scalar(
                'Return', float(np.sum(rewards)), global_step=n_cum_steps)

        if episodic_update:
            # one epoch is exactly one episode
            break

    return trace, n_cum_steps


def train_test_split(output_dir='data', trace_path='trace.csv',
                     metadata_path='metadata.json', min_train_steps=5000):
    """Split a trace into a training and test data sets

    These data sets will be used by ramp-test when training and testing the
    submissions. The split is done such that one episode is not split in two.
    The returned training and test data sets only contain whole episodes.

    Parameters
    ----------
    data_dir : string
        Directory of the trace and where the training and test sets are saved.
    trace_path : string
        Name of the trace csv file.
    metadata : string or dict
        Name of the metadata filename or dictionary.
    n_episodes_train : int
        Number of episodes to put in the training set. The rest of the episodes
        are put in the test set.
    """
    metadata = get_metadata_dictionary(metadata_path)

    trace_df = read_data_with_metadata(trace_path, metadata)
    trace_df = preprocess_time(trace_df, metadata)

    # number of samples to use in the training set
    restart_name = metadata['restart_name']
    restarts = trace_df[restart_name].to_numpy()
    episode_starts = np.where(restarts)[0]
    # get the actual total number of steps at the beginning of each episode by
    # substracting the step with NaN values.
    n_total_steps_episode_starts = (episode_starts -
                                    np.arange(len(episode_starts)))
    first_test_episode = np.where(
        n_total_steps_episode_starts >= min_train_steps)[0][0]
    first_test_ind = episode_starts[first_test_episode]

    train_trace_df = trace_df.iloc[:first_test_ind]
    test_trace_df = trace_df.iloc[first_test_ind:]

    train_trace_df.to_csv(os.path.join(output_dir, 'X_train.csv'))
    test_trace_df.to_csv(os.path.join(output_dir, 'X_test.csv'))


def get_seed_dirs(submission, agent, verbose=False):
    """Return the list of seed folders."""
    path = pathlib.Path('submissions') / submission / 'mbrl_outputs' / agent
    seed_dirs = list(path.glob('seed*'))
    if len(seed_dirs) == 0 and verbose:
        print(f'{path} not found or empty')
    return seed_dirs


def get_seed_dirs_hyperopt(submission, agent, extra='', verbose=False):
    """Return the list of seed folders."""
    path = pathlib.Path('submissions') / submission / 'mbrl_outputs' / agent / extra
    seed_dirs = list(path.glob('seed*'))
    if len(seed_dirs) == 0 and verbose:
        print(f'{path} not found or empty')
    return seed_dirs


def get_trace_df(seed_dir, verbose=False):
    """Reading and assembling trace files.

    Unavailable files are silently ignored

    Parameters
    ----------
    seed_dir : str
        folder name where the epochs are found
    verbose : boolean
        if true, unloadable traces are printed
    Returns
    -------
    pandas DataFrame (not ordered by epoch id)
    """
    metadata_path = os.path.join('data', 'metadata.json')
    metadata = get_metadata_dictionary(metadata_path)
    trace_paths = natsorted(list(seed_dir.glob('epoch_*/trace.csv')), key=str)
    trace_dfs = []
    for trace_path in trace_paths:
        try:
            trace_df = read_data_with_metadata(trace_path, metadata)
            trace_dfs.append(trace_df)
        except FileNotFoundError:
            if verbose:
                print('{} not found.'.format(trace_path))
    if len(trace_dfs) == 0:
        return None
    all_traces = pd.concat(trace_dfs, axis=0).reset_index(drop=True)
    all_traces = preprocess_time(all_traces, metadata)
    trace_df = all_traces.dropna().reset_index(drop=True).reset_index(drop=True)
    return trace_df
