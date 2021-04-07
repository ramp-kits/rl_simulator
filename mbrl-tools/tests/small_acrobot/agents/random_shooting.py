import copy

import numpy as np

from joblib import Parallel, delayed

from sklearn.utils.validation import check_random_state

N_PARTICLES = 1
N_ACTION_SEQUENCES = 5
PLANNING_HORIZON = 2
N_JOBS = 1

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
        if hasattr(self.env, 'history'):
            self.env.add_observation_to_history(observations, restart)

        if self.random_action:
            action = self.np_random.randint(N_ACTIONS)
        else:
            # recover current state/history of the environment
            if hasattr(self.env, 'history'):
                initial_state = copy.deepcopy(self.env.history)
            else:  # real system
                initial_state = copy.deepcopy(self.env.get_state())

            action_sequences = self.np_random.randint(
                N_ACTIONS, size=(N_ACTION_SEQUENCES, PLANNING_HORIZON))

            def _parallel_func(action_sequence, env, initial_state):
                returns = np.zeros(N_PARTICLES)
                for p in range(N_PARTICLES):
                    if hasattr(env, 'history'):
                        env.history = initial_state
                    else:
                        env.set_state(initial_state)

                    for a, action in enumerate(action_sequence):
                        _, reward, _, _ = env.step(action)
                        returns[p] += (self.gamma ** a * reward)

                return returns

            all_returns = Parallel(n_jobs=N_JOBS, verbose=1)(
                delayed(_parallel_func)(
                    action_sequence,
                    self.env,
                    initial_state)
                for action_sequence in action_sequences)

            all_returns = np.array(all_returns)
            returns = np.mean(all_returns, axis=1)
            returns_argmax = np.argmax(returns)
            best_action_sequence = action_sequences[returns_argmax]
            action = best_action_sequence[0]

            # put env back to its initial state
            if hasattr(self.env, 'history'):
                self.env.history = initial_state
            else:
                self.env.set_state(initial_state)

        if hasattr(self.env, 'history'):
            self.env.add_action_to_history(action)

        return action
