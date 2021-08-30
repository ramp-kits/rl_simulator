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

        self.dynamic_reset = False
        self.real_states_history = []

    def seed(self, seed=None):
        """Same as parent method but passing a RandomState instance is allowed.
        """
        self.np_random = check_random_state(seed)
        return [seed]

    def reset(self):
        """Same as parent method but resetting the number of elapsed steps."""
        if self.dynamic_reset:
            observations = self.real_states_history[
                self.np_random.choice(len(self.real_states_history))]
            self._elapsed_steps = 0
            return observations
        else:
            observations = super(Env, self).reset()
            self._elapsed_steps = 0
            return observations

    def step(self, action):
        """Same as parent method but different reward.
        We also consider that the task is never done.
        """
        observations, _, _, info = super(Env, self).step(action)
        self._elapsed_steps += 1
        reward = reward_func(np.r_[observations, action].reshape(1, -1))[0]
        # using >= in case we need the info when planning with the real env
        done = (self._elapsed_steps >= self.max_episode_steps)
        return observations, reward, int(done), info

    def set_state(self, state_dict):
        """Set state of the environment."""
        self._elapsed_steps = state_dict['_elapsed_steps']
        self.state = np.r_[state_dict['qpos'], state_dict['qvel']]

    def get_state(self):
        """Get state of the environement."""
        state = self.state
        state_dict = {
            'qpos': state[:2],
            'qvel': state[2:],
            '_elapsed_steps': self._elapsed_steps,
        }
        return state_dict

    def get_numpy_state(self):
        """Get the state numpy array from the environment."""
        state_dict = self.get_state()
        return np.r_[state_dict['qpos'], state_dict['qvel']]

    def set_numpy_state(self, numpy_state):
        """Set the state from a numpy array.

        Note that the _elapsed_steps attribute is reset to 0.
        """
        self.state = numpy_state
        self._elapsed_steps = 0
