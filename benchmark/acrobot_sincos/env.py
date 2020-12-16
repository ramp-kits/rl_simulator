import numpy as np
from sklearn.utils.validation import check_random_state
from rampwf.utils.importing import import_module_from_source

from gym.envs.classic_control import AcrobotEnv

reward_module = import_module_from_source(
    'reward_function.py', 'reward_function')
reward_func = reward_module.reward_func


class Env(AcrobotEnv):
    """Open AI Gym acrobot env with reward of benchmark paper.

    We also provide a seed method accepting instances of
    numpy.random.RandomState.
    """
    def __init__(self, max_episode_steps=200):
        self.max_episode_steps = max_episode_steps
        super(Env, self).__init__()

    def seed(self, seed=None):
        """Same as parent method but passing a RandomState instance is allowed.
        """
        self.np_random = check_random_state(seed)
        return [seed]

    def reset(self):
        """Same as parent method but resetting the number of elapsed steps."""
        observations = super(Env, self).reset()
        self._elapsed_steps = 0
        return observations

    def step(self, action):
        """Same as parent method but different reward.
        We also consider that the task is never done.
        """
        observations, _, _, info = super(Env, self).step(action)
        self._elapsed_steps += 1
        reward = reward_func(np.r_[observations, action])
        done = (self._elapsed_steps == self.max_episode_steps)
        return observations, reward, int(done), info

    def set_history(self, observation):
        """Set state of the environment.

        Used for compatibility with model environment."""
        cos_theta_1 = observation[0]
        sin_theta_1 = observation[1]
        cos_theta_2 = observation[2]
        sin_theta_2 = observation[3]
        theta_dot_1 = observation[4]
        theta_dot_2 = observation[5]
        theta_1 = np.arctan2(sin_theta_1, cos_theta_1)
        theta_2 = np.arctan2(sin_theta_2, cos_theta_2)
        self.state = np.r_[theta_1, theta_2, theta_dot_1, theta_dot_2]
