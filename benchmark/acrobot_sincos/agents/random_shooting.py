import numpy as np

from sklearn.utils.validation import check_random_state

N_PARTICLES = 5
N_ACTION_SEQUENCES = 100
PLANNING_HORIZON = 10

N_ACTIONS = 3


class Agent:
    """Random shooting.

    Parameters
    ----------
    env : gym environment
        Environment with which to run the random shooting.
    epoch_output_dir : string
        Path of the output directory of the current epoch. Can be used to save
        results.
    epsilon : float
        Value of epsilon for the epsilon-greedy exploration. Set to None if
        not epsilon-greedy not used.
    gamma : float
        Discount factor.
    random_action : bool
        Whether to draw actions at random.
    seed : int
        Seed of the RNG.
    """

    def __init__(self, env, epoch_output_dir,
                 epsilon=None, gamma=1, random_action=False,
                 seed=None):

        self.seed(seed)
        self.epoch_output_dir = epoch_output_dir
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_action = random_action

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
            restart_vec = np.array(
                [restart] * N_ACTION_SEQUENCES * N_PARTICLES)
            restart_vec = restart_vec.reshape(-1, 1)
            self.env.add_observations_to_history(observation_vec, restart_vec)

            action_sequences = self.np_random.randint(
                N_ACTIONS, size=(N_ACTION_SEQUENCES, PLANNING_HORIZON))
            action_sequences_reps = np.repeat(
                action_sequences, N_PARTICLES, axis=0)

            all_returns = np.zeros(N_ACTION_SEQUENCES * N_PARTICLES)
            for horizon in range(PLANNING_HORIZON):
                actions = action_sequences_reps[:, horizon].reshape(-1, 1)
                _, rewards, _, _ = self.env.step(actions)
                all_returns += (self.gamma ** horizon * rewards)

            all_returns = all_returns.reshape(N_ACTION_SEQUENCES, N_PARTICLES)

            returns = np.mean(all_returns, axis=1)
            returns_argmax = np.argmax(returns)
            best_action_sequence = action_sequences[returns_argmax]
            action = best_action_sequence[0]

        return action
