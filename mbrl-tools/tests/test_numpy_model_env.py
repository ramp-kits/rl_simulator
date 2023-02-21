import os

import cloudpickle

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
import pandas as pd

from mbrltools.numpy_model_env import make_model_env_class

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
    # we use None for the the model as we don't use it
    # here. However the same seeds should be used.
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import numpy_problem
    from env import Env
    ModelEnv = make_model_env_class(Env)

    seed = 1
    acrobot = Env()
    acrobot.seed(seed)
    acrobot_observation = acrobot.reset()

    model_env = ModelEnv(
        submission_path=None, problem_module=numpy_problem, reward_func=None,
        metadata=numpy_problem.metadata, output_dir=None, seed=seed)
    model_env_observation = model_env.reset()
    assert_array_almost_equal(acrobot_observation, model_env_observation)

    # check that prev_observations was correctly set
    prev_observations = model_env.prev_observations
    assert_array_almost_equal(
        prev_observations, model_env_observation.reshape(1, -1))


def test_step(monkeypatch):
    # check observations and rewards.
    # check also that the history is updated with the observed data
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import numpy_problem
    from env import Env
    ModelEnv = make_model_env_class(Env)

    metadata = numpy_problem.metadata

    # define a dummy step function used as a workflow step function and a dummy
    # reward function for the test
    def _workflow_step(model, X, random_state=None):
        action = X[0, -len(metadata['action'])-1:-1]  # -1 because of restart
        observation = np.full(shape=(1, 4), fill_value=action ** 2)
        return observation

    def _reward_func(observables):
        # observables contains observations and actions, it is important to
        # have a function of both the observations and actions for the tests.
        observations = observables[:, :4]
        action = observables[:, 4]
        return np.sum(observations) + action ** 2

    seed = 1
    model_env = ModelEnv(
        submission_path=None, problem_module=numpy_problem, reward_func=_reward_func,
        metadata=metadata, output_dir=None, seed=seed)
    # change workflow step
    model_env.workflow_step = _workflow_step
    model_env.trained_model = ''  # don't set it to None otherwise this raises an error
    actions = np.array([0, 1])

    observation = model_env.reset()
    for action in actions:
        observation, reward_1, done, _ = model_env.step(action)
        assert_array_equal(
            observation, np.full(shape=(1, 4), fill_value=action ** 2))
        reward_2 = _reward_func(
            np.concatenate((observation, [[action]]), axis=1))
        assert reward_1 == reward_2
        assert done == 0


def test_pickle(monkeypatch, tmp_path, create_random_trace):
    # check that a ModelEnv instance can be pickled. this is important when
    # using multiprocessing
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import numpy_problem
    from env import Env
    ModelEnv = make_model_env_class(Env)

    # save random initial trace used to train the model
    create_random_trace(system_env_object=Env, n_action_features=1,
                        metadata=numpy_problem.metadata, path_dir=tmp_path)
    submission_path = os.path.join('submissions', 'numpy_dummy_kit')

    model_env = ModelEnv(
        submission_path=submission_path, problem_module=numpy_problem,
        reward_func=None, metadata=numpy_problem.metadata, output_dir=tmp_path,
        seed=0)
    model_env.train_model(epoch=0)

    model_env_pkl = cloudpickle.dumps(model_env)
    del model_env
    cloudpickle.loads(model_env_pkl)
