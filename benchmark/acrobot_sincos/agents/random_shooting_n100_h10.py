import os
import numpy as np

from sklearn.utils.validation import check_random_state

default_config = {
    'N_PARTICLES': 1,
    'N_ACTION_SEQUENCES': 100,
    'PLANNING_HORIZON': 10,
}

N_ACTIONS = 3
GAMMA = 1


class Agent:
    """Random shooting.

    Parameters
    ----------
    env : gym environment
        Environment with which to run the random shooting.
    output_dir : string
        Path of the output directory. Can be used to save
        results.
    eval_env :
        Real system environment if needed to evaluate agent on it.
    eval_model_env :
        Model environment if needed to evaluate agent on it.
    planning_env :
        Model environment if needed to plan.
    config : dict
        Hyperparameters. If None default_config is used.
    metadata : dict
        Metadata.
    random_action : bool
        Whether to draw actions at random.
    seed : int
        Seed of the RNG.
    epoch : int
        First epoch to run.
    """

    def __init__(self, env, output_dir,
                 random_action=False, config=None,
                 eval_env=None, eval_model_env=None,
                 planning_env=None,
                 metadata=None,
                 seed=None, epoch=0):

        self.seed(seed)
        self._epoch = epoch
        self.output_dir = output_dir
        self.epoch_output_dir = os.path.join(self.output_dir, f'epoch_{epoch}')
        self.env = env
        self.planning_env = planning_env
        self.planning_env.reset()  # needed because of SB3

        self.planning_env._max_episode_steps = False

        self.random_action = random_action

        if config is not None:
            for hyperopt in default_config.keys():
                if hyperopt in config.keys():
                    default_config[hyperopt] = config[hyperopt]

        print(
            "Agent hyperparameters: "
            f"{[str(key) + '=' + str(value) for key, value in default_config.items()]}"
        )

        global N_PARTICLES, N_ACTION_SEQUENCES, PLANNING_HORIZON
        N_PARTICLES = default_config['N_PARTICLES']
        N_ACTION_SEQUENCES = default_config['N_ACTION_SEQUENCES']
        PLANNING_HORIZON = default_config['PLANNING_HORIZON']

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, new_epoch):
        self._epoch = new_epoch
        self.epoch_output_dir = os.path.join(self.output_dir, f'epoch_{self._epoch}')

    def seed(self, seed=None):
        # seed for numpy
        self.np_random = check_random_state(seed)
        return [seed]

    def act(self, observations, restart):
        """Return the action to take given the observations.

        Parameters
        ----------
        observations : array, shape (1, n_features)
            Observations
        restart : int
            Whether the observation is the first of an episode.

        Returns
        -------
        action : int
            The action to take.
        """
        if self.random_action:
            action = self.np_random.randint(N_ACTIONS)
        else:
            # duplicate observations and restart to leverage the vectorized
            # sampling of the model
            observation_vec = np.tile(
                observations, (N_ACTION_SEQUENCES * N_PARTICLES, 1))
            restart_vec = np.array([restart] * N_ACTION_SEQUENCES * N_PARTICLES)
            restart_vec = restart_vec.reshape(-1, 1)
            self.planning_env.add_observations_to_history(observation_vec, restart_vec)
            self.planning_env._elapsed_steps = 0

            action_sequences = self.np_random.randint(
                N_ACTIONS, size=(N_ACTION_SEQUENCES, PLANNING_HORIZON))
            action_sequences_reps = np.repeat(
                action_sequences, N_PARTICLES, axis=0)

            all_returns = np.zeros(N_ACTION_SEQUENCES * N_PARTICLES)
            for horizon in range(PLANNING_HORIZON):
                actions = action_sequences_reps[:, horizon].reshape(-1, 1)
                _, rewards, _, _ = self.planning_env.step(actions)
                all_returns += (GAMMA ** horizon * rewards)

            all_returns = all_returns.reshape(N_ACTION_SEQUENCES, N_PARTICLES)

            returns = np.mean(all_returns, axis=1)
            returns_argmax = np.argmax(returns)
            best_action_sequence = action_sequences[returns_argmax]
            action = best_action_sequence[0]

        return action
