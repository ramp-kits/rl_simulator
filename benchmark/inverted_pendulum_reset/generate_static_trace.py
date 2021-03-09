"""
Generate random trajectories for the inverted pendulum environment and split
them into a train and test datasets than are used to train and test the models.
"""
import os

import pandas as pd

from rampwf.utils.importing import import_module_from_source

from mbrltools.data_processing import rollout
from mbrltools.data_processing import train_test_split
from mbrltools.data_processing import get_metadata_dictionary

env = import_module_from_source('env.py', 'env')
Env = env.Env

min_steps = 25_000
trace_filename = 'trace.csv'
output_dir = 'data'
trace_path = os.path.join(output_dir, trace_filename)

metadata_path = os.path.join('data', 'metadata.json')
metadata = get_metadata_dictionary(metadata_path)
observation_names = metadata["observation"]
action_names = metadata["action"]
restart_name = metadata["restart_name"]
reward_name = metadata["reward"]

header = (
    observation_names + action_names + reward_name +
    [restart_name] + ['epoch_id'])

env = Env()
SEED = 32
env.seed(SEED)
env.action_space.np_random.seed(SEED)

trace = rollout(env, len(action_names), epoch=0, min_epoch_steps=min_steps)
trace_df = pd.DataFrame(data=trace, columns=header)
trace_df.to_csv(trace_path, index=False)

train_test_split(
    output_dir=output_dir,
    trace_path=trace_path,
    metadata_path=os.path.join('data', 'metadata.json'),
    min_train_steps=5000)
