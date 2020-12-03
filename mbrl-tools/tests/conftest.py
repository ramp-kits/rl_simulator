import os

import pytest

import pandas as pd

from mbrltools.data_processing import rollout


@pytest.fixture
def create_random_trace():

    def _create_random_trace(system_env_object, n_action_features,
                             metadata, path_dir):
        # create random initial trace and save it as a csv file
        env = system_env_object()
        env.seed(0)
        trace = rollout(system_env=env, n_action_features=n_action_features,
                        epoch=0, min_epoch_steps=10, agent=None)

        observation_names = metadata["observation"]
        action_names = metadata["action"]
        restart_name = metadata["restart_name"]
        reward_name = metadata["reward"]
        trace_header = (
            observation_names + action_names + reward_name +
            [restart_name] + ['epoch_id'])
        trace_df = pd.DataFrame(data=trace, columns=trace_header)
        trace_dir = os.path.join(path_dir, 'epoch_0')
        os.makedirs(trace_dir)
        trace_df.to_csv(os.path.join(trace_dir, 'trace.csv'), index=False)

    return _create_random_trace
