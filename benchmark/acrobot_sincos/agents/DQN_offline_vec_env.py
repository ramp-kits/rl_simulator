import os

import numpy as np

from sklearn.utils.validation import check_random_state

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

N_ACTIONS = 3

# best config for 10 envs
GAMMA = 0.99
TAU = 0.5
LEARNING_RATE = 0.001
EXPLORATION_FINAL_EPS = 0.01
EXPLORATION_FRACTION = 0.1
BUFFER_SIZE = 10_000
TRAIN_FREQ = 4
BATCH_SIZE = 128
TARGET_UPDATE_INTERVAL = 1000
TOTAL_TIMESTEPS = 2.5e6
LEARNING_STARTS = 1000
POLICY_KWARGS = dict(net_arch=[256, 256])
GRADIENT_STEPS = 10
# we use 10 envs to train the DQN so to log 100 times the performance we need to
# use TOTAL_TIMESTEPS // (100 times * 10 envs)
EVAL_FREQ = TOTAL_TIMESTEPS // 1000


class ModelEnvEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    Modifying the EvalCallback from sb3 to be able to have tensorboard logs with
    different names to not conflict with the evaluation environment passed to
    create_eval_env when instantiating the agent or passed to eval_env in the learn
    method. This is done by using different logging names in the _on_step method.

    The performance are logged under eval_model_env in tensorboard.
    """

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Model Eval num_timesteps={self.num_timesteps}, " f"Model episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Model Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval_model_env/mean_reward", float(mean_reward))
            self.logger.record("eval_model_env/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Model Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval_model_env/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time_model_env/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


class Agent:
    """DQN.

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
                 random_action=False, config=None, eval_env=None, eval_model_env=None,
                 planning_env=None,
                 metadata=None,
                 seed=None, epoch=0):

        self.seed(seed)
        self._epoch = epoch
        self.output_dir = output_dir
        self.epoch_output_dir = os.path.join(self.output_dir, f'epoch_{epoch}')
        self.env = env
        self.eval_env = eval_env
        self.eval_model_env = eval_model_env

        self.random_action = random_action

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
        observations : array, shape (n_features)
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
            return action

        if restart:
            self.dqn = DQN(
                "MlpPolicy", self.env,
                learning_starts=LEARNING_STARTS,
                exploration_final_eps=EXPLORATION_FINAL_EPS,
                exploration_fraction=EXPLORATION_FRACTION,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                train_freq=TRAIN_FREQ,
                target_update_interval=TARGET_UPDATE_INTERVAL,
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log=os.path.join(self.epoch_output_dir, 'tensorboard'),
            )

            if self.eval_model_env is not None:
                callback = ModelEnvEvalCallback(
                    self.eval_model_env, n_eval_episodes=1, eval_freq=EVAL_FREQ,
                    log_path=os.path.join(self.epoch_output_dir, 'model_eval'),
                    deterministic=True)
            else:
                callback = None

            # the eval_env is the real system and will be reset by sb3
            self.dqn.learn(
                callback=callback,
                total_timesteps=TOTAL_TIMESTEPS,
                log_interval=1,
                reset_num_timesteps=False,
                eval_env=self.eval_env, eval_freq=EVAL_FREQ,
                n_eval_episodes=1,
                eval_log_path=os.path.join(self.epoch_output_dir, 'system_eval')
                )

        action, _ = self.dqn.predict(observations, deterministic=True)

        return action

    def __getstate__(self):
        """Only use pickle on what can be pickled.

        For the rest we use other strategies"""
        state = self.__dict__.copy()

        # the dqn agent cannot be save with pickle because of tensorboard.
        # we could either not use tensorboard when we want to save
        # the dqn guide or we resort to the save/load method of sb3.
        # we choose the latter.
        if hasattr(self, 'dqn'):
            del state['dqn']
            self.dqn.save(os.path.join(self.epoch_output_dir, "dqn_sb3"))

        # we use the VecMonitor wrapper for following the training of the
        # dqn guide but this makes the env non pickable. We thus only
        # pickle the underlying envs and rewrapped them when setting the
        # state of the object again.
        state['venv'] = self.env.venv
        state['eval_model_venv'] = self.eval_model_env.venv
        del state['env']
        del state['eval_model_env']

        return state

    def __setstate__(self, state):
        """We only use pickle on what can be pickled.

        See __getstate__ for more info."""
        state['env'] = VecMonitor(state['venv'])
        state['eval_model_env'] = VecMonitor(state['eval_model_venv'])
        del state['venv']
        del state['eval_model_venv']
        self.__dict__.update(state)
        # the epoch_output_dir is set thanks to the update of the
        # dict
        if os.path.exists(os.path.join(self.epoch_output_dir, "dqn_sb3.zip")):
            dqn = DQN.load(os.path.join(self.epoch_output_dir, "dqn_sb3"))
            self.__dict__['dqn'] = dqn
