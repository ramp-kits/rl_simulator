import os

import numpy as np
import pandas as pd

import rampwf

from .data_processing import read_data_with_metadata
from .data_processing import preprocess_time

from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def make_model_env_class(system_env_object):
    """Make the ModelEnv class dynamically inherit from the system env class.

    Parameters
    ----------
    system_env_object : object
        AI Gym environment of the system.

    Returns
    -------
    ModelEnv : object
        See docstring of the ModelEnv class for more details.
    """

    class ModelEnv(VecEnv):
        """Open AI gym env.

        Environment associated to the learnt model of a model based RL
        strategy. See the docstring of the workflow step method for the
        description of the history.

        Parameters
        ----------
        submission_path : string
            Path of the submission.
        problem_module : object
            Problem module.
        reward_func : function
            Function taking as input observations and returning as output the
            associated reward. Can also depend on the taken actions.
        metadata : dictionary
            Providing the names of observations, actions and restart variables.
            Associated keys respectively are 'observation', 'action' and
            'restart_name'.
        output_dir : string
            Path of the output directory, where to find the data to train the
            model.
        seed : int
            Seed of the RNG used for this environment.
        termination_func : function
            Function taking as input observations and returning whether we should
            terminate.
        """

        def __init__(self, submission_path, problem_module, reward_func,
                     metadata, output_dir, partial_fit=False, seed=None,
                     save_model=True,
                     num_envs=1,
                     termination_func=None):

            self.num_envs = num_envs

            # get needed attributes from parent class. we create an instance
            # because for mujoco env calling super.__init__ would call
            # self.step and thus use the step of this class instead of the step
            # of the mujoco env
            self.system_env = system_env_object()
            self.observation_space = self.system_env.observation_space
            self.action_space = self.system_env.action_space
            self.system_env.seed(seed)

            self.submission_path = submission_path
            self.reward_func = reward_func
            self.termination_func = termination_func
            self.metadata = metadata
            self.output_dir = output_dir
            self.partial_fit = partial_fit
            self.save_model = save_model

            # only storing needed problem_module attributes as problem_module
            # can be problematic to pickle
            self.n_burn_in = problem_module._n_burn_in
            self.workflow_step = problem_module.workflow.step
            self.get_train_data = problem_module.get_train_data
            self.train_submission = problem_module.workflow.train_submission
            self._get_column_names(metadata)

            self.trained_model = None

            # for short rollouts performed with the model from real
            # observations
            self.dynamic_reset = False
            self.real_states_history = []

            self._max_episode_steps = self.system_env.max_episode_steps
            self._elapsed_steps = 0

        def _get_column_names(self, metadata):
            self.action_names = metadata['action']
            self.observation_names = metadata['observation']
            self.restart_name = metadata['restart_name']

        def add_observations_to_history(self, observations, restart):
            """Update history with the new given observation

            Parameters
            ----------
            observations : array, shape (n_samples, n_features)
                Observations.
            restarts : array, shape (n_samples, 1)
                Whether the observation is the first of an episode. This is
                used to know the history the model can use.
            """
            n_samples = observations.shape[0]

            # we do not yet support the vectorized environment if
            # n_burn_in >= 1 as this requires a different history array for
            # each sample. waiting for when we need it, we do not use
            # n_burn_in >= 1 for now.
            if n_samples > 1 and self.n_burn_in >= 1:
                raise ValueError('When n_burn_in > 1, the passed observations'
                                 ' and restart arrays must only have 1 sample')
            if n_samples != restart.shape[0]:
                raise ValueError('observations and restart arrays must have'
                                 ' the same number of samples/rows.')

            if self.n_burn_in >= 1 and restart[0]:
                # reset history
                # because of the check above observations only contains one
                # sample and restart as well
                times = np.array([[0]])
                self.history = self._build_new_history(
                    times, observations, restart)
            else:
                if self.n_burn_in >= 1:
                    times = self.history.index[-1] + 1
                    times = np.array([times]).reshape(1, -1)
                else:
                    times = np.zeros((n_samples, 1))

                history_df = self._build_new_history(
                    times, observations, restart)

                if self.n_burn_in >= 1:
                    # we concatenate with previous history
                    history_df = pd.concat([self.history, history_df], axis=0)
                    history_df = history_df.iloc[-(self.n_burn_in + 1):]

                self.history = history_df

        def _build_new_history(self, times, observations, restart):
            """Build new history.

            To be used as new history or to be appended to previous history
            if n_burn_in >= 1.

            Parameters
            ----------
            times : array, shape (n_samples, 1)
                Times to use for the new history samples.
            observations : array, shape (n_samples, n_features)
                Observations to be put in the new history
            restart : array, shape (n_samples, 1)
                Restarts.

            Returns
            -------
            history_df : pandas dataframe, shape (n_samples, n_features + 2)
                New history.
            """
            n_samples = observations.shape[0]

            history_col_names = (
                    ['time'] + self.observation_names + self.action_names +
                    [self.restart_name])

            # the unknown next actions are set to NaN for now and will be
            # replaced by the actions of the next call to step.
            n_action_features = len(self.action_names)
            nan_actions = np.full(
                (n_samples, n_action_features), np.nan)

            history = np.concatenate(
                (times, observations, nan_actions, restart),
                axis=1)
            history_df = (pd.DataFrame(data=history, columns=history_col_names)
                          .set_index('time'))

            return history_df

        def add_action_to_history(self, actions):
            """Update history with the given action.

            Add this action to the last observation of the history.

            Parameters
            ----------
            actions : array, shape (n_samples, n_action_features)
                Action.
            """
            n_samples = actions.shape[0]
            action_col_ind = self.history.columns.get_indexer(
                self.action_names)
            self.history.iloc[-n_samples:, action_col_ind] = actions

        def reset(self):
            """Reset method of the environment.

            The history of the model is also reset.

            Returns
            -------
            observation : numpy array, shape (n_observations,)
                The passed observation if not None or a new observation.
            """
            if self.dynamic_reset:
                observations = self.real_states_history[
                    self.np_random.choice(
                        len(self.real_states_history),
                        size=self.num_envs)]
            else:
                observations = np.array([
                    self.system_env.reset() for _ in range(self.num_envs)])

            self.add_observations_to_history(
                observations, np.ones((self.num_envs, 1)))
            self._elapsed_steps = 0

            return observations

        def _workflow_step(self, history, seed=None):
            """Compute step from history history.

            Parameters
            ----------
            history : pandas DataFrame
                History. Contains past data and the new action.

            seed : int
                Seed of the RNG used to sample the new observation.

            Returns
            -------
            observation : pandas DataFrame
                The sampled observation.
            """
            if self.trained_model is None:
                raise ValueError('You need a trained model to apply the step method')
            observations = self.workflow_step(
                self.trained_model, history, random_state=seed)
            return observations

        def train_model(self, epoch):
            """Update model with collected data.

            Parameters
            ----------
            epoch : int
                Epoch of the main loop. Used to know how many traces should be
                used to update the model.
            """
            output_dir = self.output_dir
            metadata = self.metadata

            # get all previous traces, concatenate them and update model
            trace_paths = [os.path.join(output_dir, f'epoch_{i}', 'trace.csv')
                           for i in range(epoch + 1)]
            trace_dfs = []
            for trace_path in trace_paths:
                trace_df = read_data_with_metadata(trace_path, metadata)
                trace_dfs.append(trace_df)
            all_traces = pd.concat(trace_dfs, axis=0).reset_index(drop=True)
            all_traces = preprocess_time(all_traces, metadata)

            epoch_output_dir = os.path.join(output_dir, f'epoch_{epoch}')
            training_data_dir = os.path.join(epoch_output_dir, 'data')
            if not os.path.exists(training_data_dir):
                os.makedirs(training_data_dir)

            all_traces.to_csv(os.path.join(training_data_dir, 'X_train.csv'))
            X_train, y_train = self.get_train_data(epoch_output_dir)

            if epoch == 0 or not self.partial_fit:
                trained_model = self.train_submission(
                    self.submission_path, X_train, y_train)
            else:
                trained_model = self.train_submission(
                    self.submission_path, X_train, y_train,
                    prev_trained_model=self.trained_model)

            # saving trained model, will raise an error if a model cannot be
            # pickled
            if self.save_model:
                rampwf.utils.pickle_trained_model(
                    epoch_output_dir, trained_model, is_silent=False)

            self.trained_model = trained_model

        def __getstate__(self):
            """Needed to override this method of the parent class.

            Sometimes the parent class, the system environment object,
            implements its own __getstate__ method and makes the copy of
            the ModelEnv object fail."""
            state = self.__dict__.copy()
            return state

        def __setstate__(self, state):
            """Needed to override this method of the parent class.

            Sometimes the parent class, the system environment object,
            implements its own __setstate__ method and makes the copy of
            the ModelEnv object fail."""
            self.__dict__.update(state)

        # all the following methods are abstract methods of VecEnv
        def close(self) -> None:
            return super().close()

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            return super().env_method(
                method_name, *method_args, indices=indices, **method_kwargs)

        def get_attr(self, attr_name, indices=None):
            return super().get_attr(attr_name, indices)

        def seed(self, seed=None):
            return super().seed(seed)

        def set_attr(self, attr_name, value, indices=None):
            return super().set_attr(attr_name, value, indices)

        def step_async(self, actions):
            self.actions = actions

        def step_wait(self):
            """Step function of the model environment.

            The history of the environment is used by the model for the
            dynamics prediction and updated at each step with the given action
            and returned observations.
            Note that done is returned for compatibility but is always set to
            0 as we do not consider early terminations when using the model.

            Parameters
            ----------
            actions : int or numpy array, shape (n_samples, n_action_features
            or (n_action_features,)
                The actions to be taken. Can be an int if action_space is
                of Discrete type. If actions is a 1D array it is assumed
                that it contains one sample. Allowing to pass int or a 1D array
                is needed for compatibility with gym environments, for instance
                when training a model-free agent with the model environment.

            Returns
            -------
            observation : numpy array, shape (n_samples, n_features)
                The sampled observations.

            reward : numpy array, shape (n_samples,)
                Reward computed from the taken action and the obtained
                observations.

            done : numpy array, shape (n_samples)
                An array of zeros is always returned.

            info : dict
                Empty dict, used for compatibility with AI Gym API.
            """
            actions = self.actions
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)

            n_samples = actions.shape[0]
            if n_samples > 1 and self.n_burn_in >= 1:
                raise ValueError(
                    'When n_burn_in > 1, the passed actions array must only '
                    'have 1 sample')
            self.add_action_to_history(actions)

            observations = self._workflow_step(self.history, seed=None)
            observations = observations.to_numpy()
            observations = np.clip(
                observations,
                self.observation_space.low,
                self.observation_space.high)
            self._elapsed_steps += 1

            rewards = self.reward_func(np.concatenate((observations, actions), axis=1))

            self.add_observations_to_history(
                observations, np.zeros((n_samples, 1))
            )

            if self.termination_func is not None:
                done = self.termination_func(observations)
            else:
                done = np.zeros(n_samples).astype(bool)

            if (self._max_episode_steps and
                    self._elapsed_steps == self._max_episode_steps):
                done = np.ones(n_samples).astype(bool)

            if done.any():
                infos = [{"terminal_observation": obs} for obs in observations]
                observations = self.reset()
            else:
                done = np.zeros(n_samples).astype(bool)
                infos = [{} for _ in range(self.num_envs)]

            return observations, rewards, done, infos

    return ModelEnv
