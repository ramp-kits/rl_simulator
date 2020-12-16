import math as m


def reward_func(observations):
    """Computation of the reward from the observations of the inverted
    pendulum env.

    The observations are also the ones predicted by the model.
    For compatibility with other environments the actions should be appended to
    the inputs observations as the reward can be a function of both the actions
    and the observations.

    In the case where we want to compute the return as a measure of performance
    (for non fixed length episodes) we need the reward to be positive.
    Otherwise, if the reward is always negative, there is no incentive for the
    pole to stay up as long as possible as this will only decrease the return.

    Parameters
    ----------
    observations : array, shape (n_observations + n_actions,)
        Observations and actions. The last features are the actions.
        Note that this is the action leading to the obtained observations.

    Return
    ------
    reward : float
        Reward.
    """
    angle = observations[1]
    reward = (m.pi / 2) ** 2 - angle ** 2

    return reward
