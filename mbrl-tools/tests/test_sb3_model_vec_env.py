import os

import cloudpickle

from mbrltools.sb3_model_vec_env import make_model_env_class

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
        seed=0, num_envs=10)
    model_env.train_model(epoch=0)

    model_env_pkl = cloudpickle.dumps(model_env)
    del model_env
    cloudpickle.loads(model_env_pkl)
