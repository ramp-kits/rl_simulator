import os
import itertools
from mbrltools.model_env import make_model_env_class

import numpy as np
from numpy.testing import assert_array_equal

from mbrltools.data_processing import rollout

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


def test_rollout(monkeypatch):
    # check rollout on acrobot with short episodes and cyclic agent
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    from env import Env

    class CyclicAgent:
        def __init__(self, actions=np.array([0, 1, 2])):
            self.action_choice = itertools.cycle(actions)

        def act(self, observation, restart):
            # acrobot action space is {0, 1, 2}, we cycle over these actions.
            action = next(self.action_choice)
            return action

    epoch = 0
    max_episode_steps = 5
    n_episodes = 2
    min_epoch_steps = n_episodes * max_episode_steps  # 2 episodes in epoch

    system_env = Env(max_episode_steps=max_episode_steps)
    system_env.seed(0)

    agent = CyclicAgent()

    trace = rollout(epoch=epoch, min_epoch_steps=min_epoch_steps,
                    system_env=system_env, agent=agent, n_action_features=1,
                    episodic_update=False)

    # adding n_episodes in right side to count last observations of the
    # episodes, the ones just before the reset
    assert isinstance(trace, list)

    trace_array = np.asarray(trace)
    # columns of trace_array are observations, actions, reward, done and epoch
    assert trace_array.shape == (min_epoch_steps + n_episodes, 8)
    trace_actions = trace_array[:, 4]
    expected_actions = np.array(
        [0.,  1.,  2.,  0.,  1., np.nan,  2.,  0.,  1.,  2.,  0., np.nan])
    assert_array_equal(trace_actions, expected_actions)

    trace_restart = trace_array[:, 6]
    expected_restarts = np.array(
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert_array_equal(trace_restart, expected_restarts)

    trace_epoch = trace_array[:, 7]
    assert_array_equal(trace_epoch, np.zeros(min_epoch_steps + n_episodes))

    # check when episodic_update is set to True
    trace = rollout(epoch=epoch, min_epoch_steps=min_epoch_steps,
                    system_env=system_env, agent=agent, n_action_features=1,
                    episodic_update=True)
    # trace contains only one episode
    assert len(trace) == max_episode_steps + 1


def test_agent_with_true_dynamics(monkeypatch):
    # pass true env in the agent and check that it works as the true env
    # this is to check that we can test the agent with the true dynamics
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import problem
    from env import Env
    from reward_function import reward_func
    from agents.random_shooting import Agent

    agent_env = Env()
    agent = Agent(agent_env, epoch_output_dir=None)
    agent.env.seed(1)
    agent_observation_1 = agent.env.reset()
    agent_observation_2, agent_reward, _, _ = agent.env.step(1)

    env = Env()
    env.seed(1)
    true_observation_1 = env.reset()
    true_observation_2, true_reward, _, _ = env.step(1)

    assert_array_equal(agent_observation_1, true_observation_1)
    assert_array_equal(agent_observation_2, true_observation_2)
    assert agent_reward == true_reward


def test_model_based_agent_custom(monkeypatch, tmp_path, create_random_trace):
    # test with a dummy Agent and dummy model
    monkeypatch.chdir(small_acrobot_dir_path)
    monkeypatch.syspath_prepend(small_acrobot_dir_path)
    import problem
    from env import Env
    from reward_function import reward_func

    metadata = problem.metadata

    # save random initial trace used to train the model
    create_random_trace(system_env_object=Env, n_action_features=1,
                        metadata=metadata, path_dir=tmp_path)

    submission_path = os.path.join('submissions', 'dummy_kit')

    class DummyAgent():
        def __init__(self, env, epoch_output_dir,
                     random_action=False, seed=None):
            self.env = env
            self.env.reset()  # initialize history
            self.random_action = random_action

        def act(self, observation):
            # should return 0 if random action, 1 otherwise given the model
            if self.random_action:
                return 0
            else:
                # the model always predict 10
                observation, _, _, _ = self.env.step(1)
                if (observation == 10).all():
                    return 1
                else:
                    return 2

    for random_action, expected_action in zip([False, True], [1, 0]):
        ModelEnv = make_model_env_class(Env)
        model_env = ModelEnv(
            submission_path, problem, reward_func, metadata, tmp_path)
        model_env.train_model(epoch=0)
        dummy_agent = DummyAgent(
            model_env, None, random_action=random_action)

        assert dummy_agent.act(observation=None) == expected_action
