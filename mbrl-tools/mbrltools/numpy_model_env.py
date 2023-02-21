import os
import warnings
from pickle import PicklingError

import numpy as np
import pandas as pd
import cloudpickle

from gym.spaces import Discrete

from .data_processing import read_data_with_metadata


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

            # get needed attributes from parent class. we create an instance
            # because for mujoco env calling super.__init__ would call
            # self.step and thus use the step of this class instead of the step
            # of the mujoco env
            system_env = system_env_object()
            self.action_space = system_env.action_space
            self.observation_space = system_env.observation_space
            super(ModelEnv, self).seed(seed)

            self.submission_path = submission_path
            self.reward_func = reward_func
            self.metadata = metadata
            self.output_dir = output_dir

            # only storing needed problem_module attributes as problem_module
            # can be problematic to pickle
            self.workflow_step = problem_module.workflow.step
            self.get_train_data = problem_module.get_train_data
            self.train_submission = problem_module.workflow.train_submission

            self.trained_model = None

        def reset(self):
            """Reset method of the environment.

            The history of the model is also reset.

            Returns
            -------
            observations : numpy array, shape (n_observations,)
                The passed observation if not None or a new observation.
            """
            observations = super(ModelEnv, self).reset()
            self.prev_observations = observations.reshape(1, -1)

            return observations

        def step(self, actions):
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
            observations : numpy array, shape (n_samples, n_features)
                The sampled observations.

            rewards : numpy array, shape (n_samples,)
                Reward computed from the taken action and the obtained
                observations.

            dones : numpy array, shape (n_samples)
                An array of zeros is always returned.

            info : dict
                Empty dict, used for compatibility with AI Gym API.
            """
            if self.trained_model is None:
                raise ValueError('You need a trained model to apply the step method')

            if (isinstance(self.action_space, Discrete) and
                    not isinstance(actions, np.ndarray)):
                actions = np.array([actions]).reshape(1, -1)

            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            n_samples = actions.shape[0]
            # we also add restarts to be compatible with ramp for which we have a
            # restart column for the CV, even if it is not used in the model
            restarts = np.zeros(n_samples).reshape(-1, 1)
            model_input = np.concatenate(
                (self.prev_observations, actions, restarts), axis=1)
            observations = self.workflow_step(self.trained_model, model_input)
            rewards = self.reward_func(np.concatenate((observations, actions), axis=1))

            self.prev_observations = observations

            # we do not terminate early when using the models and thus always
            # return 0 for the done variable
            dones = np.zeros(n_samples)

            return observations, rewards, dones, {}

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

            self.trained_model = trained_model

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
