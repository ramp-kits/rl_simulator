import os

import cloudpickle

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
import pandas as pd

from mbrltools.model_env import make_model_env_class

# To be able to test the functions with the small_acrobot test example
# we use monkeypatching to set the small_acrobot/ dir as the current working
# directory which is supposed to be the case when running model-based-rl and
# thus calling the ModelEnv methods. We also add the small_acrobot/
# dir to sys.path to be able to easily import the associated modules needed
# for the sake of testing. Note that when calling model-based-rl, the root dir
# is not in sys.path so this is not completely mimicking the setup of how
# ModelEnv is used in production.
# Other solutions include:
# 1. Using the rampwf function import_module_from_source
# 2. Add __init__.py in the test folders so as to have an explicit package
# and make pytest discover this package. This would thus allow to import other
# modules.
DIR_PATH = os.path.dirname(__file__)
small_acrobot_dir_path = os.path.join(DIR_PATH, 'small_acrobot')


def test_reset(monkeypatch):
    # check that reset is similar to the base environment.
    # we use None as the workflow_step and the model as we don't use them
    # here. However the same seeds should be used.
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import problem
    from env import Env
    ModelEnv = make_model_env_class(Env)

    seed = 1
    acrobot = Env()
    acrobot.seed(seed)
    acrobot_observation = acrobot.reset()

    model_env = ModelEnv(
        submission_path=None, problem_module=problem, reward_func=None,
        metadata=problem.metadata, output_dir=None, seed=seed)
    model_env_observation = model_env.reset()
    assert_array_almost_equal(acrobot_observation, model_env_observation)

    # check that history was correctly set
    history = model_env.history.to_numpy()
    assert_array_almost_equal(
        history, np.r_[model_env_observation, np.nan, 1].reshape(1, -1))


def test_add_action_and_add_observations_to_history(monkeypatch):
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import problem
    from env import Env
    ModelEnv = make_model_env_class(Env)

    seed = 1
    rng = np.random.RandomState(seed)
    model_env = ModelEnv(
        submission_path=None, problem_module=problem, reward_func=None,
        metadata=problem.metadata, output_dir=None, seed=seed)
    rand_observation = rng.randn(1, 4)
    restart = np.array([[1]])
    model_env.add_observations_to_history(rand_observation, restart)
    history_1 = model_env.history.to_numpy()
    assert_array_almost_equal(
        history_1,
        np.concatenate([rand_observation, [[np.nan]], restart], axis=1)
    )
    action = np.array([[1]])
    model_env.add_action_to_history(action)
    history_2 = model_env.history.to_numpy()
    assert_array_almost_equal(
        history_2,
        np.concatenate([rand_observation, action, restart], axis=1)
    )


def test_step(monkeypatch):
    # check observations and rewards.
    # check also that the history is updated with the observed data
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import problem
    from env import Env
    ModelEnv = make_model_env_class(Env)

    metadata = problem.metadata

    # define a dummy step function used as a workflow step function and a dummy
    # reward function for the test
    def _workflow_step(model, X_df, random_state=None):
        action_col_num = X_df.columns.get_indexer(metadata['action'])
        action = X_df.iloc[-1, action_col_num]
        observation = pd.DataFrame(
            data=np.full(shape=(1, 4), fill_value=action ** 2),
            columns=metadata['observation'])
        return observation

    def _reward_func(observables):
        # observables contains observations and actions, it is important to
        # have a function of both the observations and actions for the tests.
        observations = observables[:, :4]
        action = observables[:, 4]
        return np.sum(observations) + action ** 2

    seed = 1
    model_env = ModelEnv(
        submission_path=None, problem_module=problem, reward_func=_reward_func,
        metadata=metadata, output_dir=None, seed=seed)
    # change workflow step and n_burn_in
    model_env.workflow_step = _workflow_step
    model_env.n_burn_in = 3  # set n_burn_in so that we have the full history
    model_env.model = None  # set model to None as needed but not used
    actions = np.array([0, 1])

    observation = model_env.reset()
    history_1 = np.full(shape=(3, 6), fill_value=np.nan)
    history_1[0, :4] = observation
    history_1[0, 5] = 1  # restart flag
    for a, action in enumerate(actions):
        history_1[a, 4] = action
        observation, reward_1, done, _ = model_env.step(action)
        assert_array_equal(
            observation, np.full(shape=(1, 4), fill_value=action ** 2))
        reward_2 = _reward_func(
            np.concatenate((observation, [[action]]), axis=1))
        assert reward_1 == reward_2
        assert done == 0
        history_1[a + 1, :4] = observation
        history_1[a + 1, 5] = 0  # restart flag

    assert_array_almost_equal(history_1, model_env.history.to_numpy())


def test_pickle(monkeypatch, tmp_path, create_random_trace):
    # check that a ModelEnv instance can be pickled. this is important when
    # using multiprocessing
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import problem
    from env import Env
    ModelEnv = make_model_env_class(Env)

    # save random initial trace used to train the model
    create_random_trace(system_env_object=Env, n_action_features=1,
                        metadata=problem.metadata, path_dir=tmp_path)
    submission_path = os.path.join('submissions', 'dummy_kit')

    model_env = ModelEnv(
        submission_path=submission_path, problem_module=problem,
        reward_func=None, metadata=problem.metadata, output_dir=tmp_path,
        seed=0)
    model_env.train_model(epoch=0)

    model_env_pkl = cloudpickle.dumps(model_env)
    del model_env
    cloudpickle.loads(model_env_pkl)
