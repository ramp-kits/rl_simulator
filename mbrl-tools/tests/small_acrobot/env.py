import os

import numpy as np
from sklearn.utils.validation import check_random_state
from rampwf.utils.importing import import_module_from_source

from gym.envs.classic_control import AcrobotEnv

dir_path = os.path.dirname(__file__)
reward_module_path = os.path.join(dir_path, 'reward_function.py')
reward_module = import_module_from_source(
    reward_module_path, 'reward_function')
reward_func = reward_module.reward_func


class Env(AcrobotEnv):
    """Modified Open AI Gym acrobot returning states instead of observations.

    We also provide a seed method accepting instances of
    numpy.random.RandomState.
    """
    def __init__(self, max_episode_steps=5):
        self.max_episode_steps = max_episode_steps
        super(Env, self).__init__()

    def seed(self, seed=None):
        """Same as parent method but passing a RandomState instance is allowed.
        """
        self.np_random = check_random_state(seed)
        return [seed]

    def reset(self):
        """Same as parent method but returns states instead of observations."""
        super(Env, self).reset()
        self._elapsed_steps = 0
        return self.state

    def step(self, action):
        """Same as parent method but returns states instead of observations.
        We also consider that the task is never done.
        """
        _, _, _, info = super(Env, self).step(action)
        self._elapsed_steps += 1
        observations = self.state
        reward = reward_func(np.r_[observations, action])
        done = (self._elapsed_steps == self.max_episode_steps)
        return observations, reward, int(done), info

    def set_history(self, observation):
        """Set state of the environment.

        Used for compatibility with model environment."""
        self.state = observation
