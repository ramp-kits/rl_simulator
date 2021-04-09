

def reward_func(observations):
    """Computation of the reward from the observations of the inverted
    pendulum env.

    The observations are also the ones predicted by the model.
    For compatibility with other environments the actions should be appended to
    the inputs observations as the reward can be a function of both the actions
    and the observations.

    Parameters
    ----------
    observations : array, shape (n_samples, n_observations + n_actions)
        Observations and actions. The last features are the actions.
        Note that this is the action leading to the obtained observations.

    Return
    ------
    reward : float
        Reward.
    """
    angle = observations[:, 1]
    reward = - angle ** 2

    return reward
