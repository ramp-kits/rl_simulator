import os
import warnings
from pickle import PicklingError

import numpy as np
import pandas as pd
import cloudpickle

from .data_processing import read_data_with_metadata
from .data_processing import preprocess_time


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

    class ModelEnv(system_env_object):
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
        """

        def __init__(self, submission_path, problem_module, reward_func,
                     metadata, output_dir, seed=None):
            self.submission_path = submission_path
            self.reward_func = reward_func
            self.metadata = metadata
            self.output_dir = output_dir
            super(ModelEnv, self).seed(seed)

            # only storing needed problem_module attributes as problem_module
            # can be problematic to pickle
            self.n_burn_in = problem_module._n_burn_in
            self.workflow_step = problem_module.workflow.step
            self.get_train_data = problem_module.get_train_data
            self.train_submission = problem_module.workflow.train_submission
            self._get_column_names(metadata)

            # set history to None, this attribute is used to check if an env
            # has an history that needs to be set
            self.history = None

        def _get_column_names(self, metadata):
            self.action_names = metadata['action']
            self.observation_names = metadata['observation']
            self.restart_name = metadata['restart_name']

        def set_history(self, observation, restart):
            """Reset the history with the given observations

            Parameters
            ----------
            observation : array, shape (n_features)
                Observations.
            restart : int
                Whether the observation is the first of an episode. This is
                used to know the history the model can use.
            """
            # reset history
            history_col_names = (['time'] + self.observation_names +
                                 self.action_names + [self.restart_name])
            n_action_features = len(self.action_names)
            # XXX we use 0 as an arbitrary start time
            # each sample of the history is made of one observation and one
            # action, the action being the one selected after the given
            # observation.
            # the unknown next action is set to NaN for now and will be
            # replaced by the action of the next call to step.
            history = np.r_[
                0, observation.ravel(), [np.nan] * n_action_features, restart]
            history = history.reshape(1, -1)
            history = (pd.DataFrame(data=history, columns=history_col_names)
                       .set_index('time'))
            self.history = history

        def add_observation_to_history(self, observation, restart):
            """Update history with the new given observation

            Parameters
            ----------
            observation : array, shape (n_features)
                Observations.
            restart : int
                Whether the observation is the first of an episode. This is
                used to know the history the model can use.
            """
            if restart:
                self.set_history(observation, restart)
            else:
                history_df = self.history
                # the action is set to NaN for now and will be updated when
                # needed
                new_sample_col_names = (
                    ['time'] + self.observation_names +
                    self.action_names + [self.restart_name])
                new_time = history_df.index[-1] + 1

                # the unknown next action is set to NaN for now and will be
                # replaced by the action of the next call to step.
                n_action_features = len(self.action_names)
                new_sample = np.r_[
                    new_time, observation,
                    [np.nan] * n_action_features,
                    restart]
                new_sample = new_sample.reshape(1, -1)
                new_sample = pd.DataFrame(
                    data=new_sample,
                    columns=new_sample_col_names).set_index('time')
                history_df = pd.concat([history_df, new_sample], axis=0)
                # keep history size less than n_burn_in + 1 samples
                self.history = history_df.iloc[-(self.n_burn_in + 1):]

        def add_action_to_history(self, action):
            """Update history with the given action.

            Add this action to the last observation of the history.

            Parameters
            ----------
            action : array
                Action.
            """
            action_col_num = self.history.columns.get_indexer(
                self.action_names)
            self.history.iloc[-1, action_col_num] = action

        def reset(self):
            """Reset method of the environment.

            The history of the model is also reset.

            Returns
            -------
            observation : numpy array, shape (n_observations,)
                The passed observation if not None or a new observation.
            """
            observation = super(ModelEnv, self).reset()
            self.set_history(observation, 1)

            return observation

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
            observations = self.workflow_step(
                self.model, history, random_state=seed)
            return observations

        def step(self, action):
            """Step function of the model environment.

            The history of the environment is used by the model for the
            dynamics prediction and updated at each step with the given action
            and returned observations.
            Note that done is returned for compatibility but is always set to
            0 as we do not consider early terminations when using the model.

            Parameters
            ----------
            action : numpy array, shape (n_action_features,)
                The action to be taken.

            Returns
            -------
            observation : numpy array, shape (n_observations,)
                The sampled observations.

            reward : float
                Reward computed from the taken action and the obtained
                observations.

            done : int
                0 is always returned.

            info : dict
                Empty dict, used for compatibility with AI Gym API.
            """
            self.add_action_to_history(action)

            observation = self._workflow_step(self.history, seed=None)
            observation = observation.to_numpy().ravel()
            reward = self.reward_func(np.r_[observation, action])

            self.add_observation_to_history(observation, 0)

            # we do not terminate early when using the models and thus always
            # return 0 for the done variable
            done = 0

            return observation, reward, done, {}

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
            trained_model = self.train_submission(
                self.submission_path, X_train, y_train)

            # saving trained model, will raise an error if a model cannot be
            # pickled
            model_filename = os.path.join(
                epoch_output_dir, 'trained_submission.pkl')
            with open(model_filename, 'wb') as f:
                try:
                    cloudpickle.dump(trained_model, f)
                except PicklingError:
                    msg = ('Using dill instead of cloudpickle to pickle '
                           'trained submission.')
                    warnings.warn(msg)
                    import dill
                    dill.dump(trained_model, f)

            self.model = trained_model

        def __getstate__(self):
            """Needed to override this method of the parent class.

            Sometimes the parent class, the system environment object,
            implements its own __getstate__ method and makes the copy of
            the ModelEnv object fail."""
            return self.__dict__.copy()

        def __setstate__(self, state):
            """Needed to override this method of the parent class.

            Sometimes the parent class, the system environment object,
            implements its own __setstate__ method and makes the copy of
            the ModelEnv object fail."""
            self.__dict__.update(state)

    return ModelEnv
