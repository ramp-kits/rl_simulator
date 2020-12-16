import numpy as np
from sklearn.utils.validation import check_random_state
from rampwf.utils.importing import import_module_from_source

from gym.envs.mujoco import InvertedPendulumEnv

reward_module = import_module_from_source(
    'reward_function.py', 'reward_function')
reward_func = reward_module.reward_func


class Env(InvertedPendulumEnv):
    """Modified Open AI Gym invertedPendulum.

    The reward is one defined in the benchmark paper.
    We also provide a seed method accepting instances of
    numpy.random.RandomState.
    """
    def __init__(self, max_episode_steps=100):
        self.max_episode_steps = max_episode_steps
        # needed because Mujoco calls step in init
        self._elapsed_steps = 0
        super(Env, self).__init__()

    def seed(self, seed=None):
        """Same as parent method but passing a RandomState instance is allowed.
        """
        self.np_random = check_random_state(seed)
        return [seed]

    def reset(self):
        """Same as parent method but returns states instead of observations."""
        observations = super(Env, self).reset()
        self._elapsed_steps = 0
        return observations

    def step(self, action):
        """Same as parent method but using benchmark paper reward"""
        observation, _, _, info = super(Env, self).step(action)
        self._elapsed_steps += 1
        reward = reward_func(np.r_[observation, action])
        done = (self._elapsed_steps == self.max_episode_steps)
        return observation, reward, done, info

    def __getstate__(self):
        """For copying and pickling.

        By default copying and pickling are handled by EzPickle from which this
        class inherits. However EzPickle is such that copying and pickling does
        not preserve the state [qpos, qvel] of the environment. We thus add the
        state of the environment

        Returns
        -------
        state_dict : dict
            The default dictionary returned by EzPickle.__getstate__ and two
            additional keys 'qpos' and 'qvel' for the state of the
            environment
        """
        state_dict = super(Env, self).__getstate__()
        state_dict['qpos'] = self.data.qpos.ravel()
        state_dict['qvel'] = self.data.qvel.ravel()
        state_dict['_elapsed_steps'] = self._elapsed_steps
        return state_dict

    def __setstate__(self, state_dict):
        """For copying and pickling.

        By default copying and pickling are handled by EzPickle from which this
        class inherits. However EzPickle is such that copying and pickling does
        not preserve the state [qpos, qvel] of the environment. We thus add the
        state of the environment

        Parameters
        ----------
        state_dict : dict
            The one returned by __getstate__.
        """
        state_dict_no_data = {key: state_dict[key] for key in state_dict
                              if key != 'qpos' and key != 'qvel'}
        super(Env, self).__setstate__(state_dict_no_data)
        self.set_state(state_dict['qpos'], state_dict['qvel'])
        self._elapsed_steps = state_dict['_elapsed_steps']
