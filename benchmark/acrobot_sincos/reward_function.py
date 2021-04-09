

def reward_func(observations):
    """Computation of the reward from the observations of the acrobot env.

    The observations are also the ones predicted by the model.
    For compatibility with other environments the actions should be appended to
    the inputs observations as the reward can be a function of both the actions
    and the observations.

    The original reward of acrobot is in [-2, 2]. We scale it to [0, 4]

    Parameters
    ----------
    observations : array, shape (n_samples, n_observations + n_actions)
        Observations and actions. The last feature is the action, which is not
        used here but put for compatibility with other environments.
        Note that this is the action leading to the obtained observations.

    Return
    ------
    reward : float
        Reward.
    """

    reward = 2 - (observations[:, 0] +
                  observations[:, 0] * observations[:, 2] -
                  observations[:, 1] * observations[:, 3])

    return reward
