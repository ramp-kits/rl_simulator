import os
import json
import click

from ray import tune
from ray.tune.logger import TBXLoggerCallback

from .model_based_rl import mbrl_run
from .data_processing import get_seed_dirs_hyperopt
from .leaderboard import mar_score

ABS_PATH = os.getcwd()


def mbrl_hyperopt(agent_name, submission,
                  n_epochs, min_epoch_steps, min_random_steps, n_epoch_episodes=None,
                  episodic_update=False,
                  data_label="",
                  model_env_module="model_env", num_envs=10,
                  seed=99999, partial_fit=False,
                  epoch_resume=None,
                  save_model=True, save_agent=True,
                  num_samples=1,
                  mar_threshold=None):
    """Main script of model based RL hyperparameter optimiziation.

    Check docstrings of model_based_rl.mbrl_run_command for arguments not listed below.

    Parameters
    ----------
    num_samples: int
        Number of samples for ray-tune.
    mar_threshold : int or None
        The number of epochs from which we suppose convergence
        If None, read from metadata
        If no metadata, set to 0.5
    """
    agent_config_file = os.path.join('agents', agent_name + '_config.json')
    with open(agent_config_file, "r") as json_file:
        config = json.load(json_file)

    config = {key: eval(value) for key, value in config.items()}

    def training_function(config, checkpoint_dir=None):
        print(f"Hyperparameters candidate: {config}")
        print("==========================================")

        os.chdir(ABS_PATH)
        extra_path = os.path.join('hyperopt', tune.get_trial_id())

        # save config in this folder
        trial_dir = os.path.join(
            'submissions', submission, 'mbrl_outputs', agent_name,
            extra_path)
        os.makedirs(trial_dir)
        given_agent_config_file = os.path.join(trial_dir, 'config.json')
        with open(given_agent_config_file, "w") as json_file:
            json.dump(config, json_file)

        mbrl_run(
            agent_name, submission,
            n_epochs, min_epoch_steps, min_random_steps, n_epoch_episodes,
            episodic_update,
            model_env_module, num_envs,
            data_label,
            seed, partial_fit,
            epoch_resume,
            save_model, save_agent,
            config=config, hyperopt=True)

        seed_dirs = get_seed_dirs_hyperopt(
            submission, agent_name, extra=extra_path, verbose=False)
        if len(seed_dirs) == 0:
            print(f'{submission}/{agent_name}/{extra_path} is not found')

        mar = mar_score(seed_dirs, threshold=mar_threshold)

        # Feed the score back back to Tune.
        tune.report(mar=mar[0])

    analysis = tune.run(
        training_function,
        config=config,
        metric="mar",
        mode="max",
        callbacks=[TBXLoggerCallback()],
        num_samples=num_samples,
    )

    best_trial = analysis.get_best_trial(metric="mar", mode="max")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial MAR: {best_trial.last_result['mar']}")


@click.command()
@click.option('--agent-name', default='random_shooting', show_default=True,
              type=click.STRING, help="Agent.")
@click.option("--submission", default="real_system", show_default=True,
              type=click.STRING,
              help="Model submission. Choose 'real_system' if you want to "
              "use the real environment.")
@click.option("--n-epochs", default=100, show_default=True, type=click.INT,
              help="The number of epochs. If the submission is not the real "
              "system, the model is updated at each epoch. If initial-trace "
              "is set to True the first epoch is assumed to be the initial "
              "trace.")
@click.option("--min-epoch-steps", default=200, show_default=True,
              type=click.INT,
              help="The minimum number of steps for each epoch given that "
              "each epoch ends by a complete episode.")
@click.option("--min-random-steps", default=None, show_default=True,
              type=click.INT,
              help="The minimum number of steps done at the first epoch"
              " with the random policy if initial-trace is set to False. "
              "If None then it is equal to min-epoch-steps.")
@click.option("--n-epoch-episodes", default=None, show_default=True,
              type=click.INT,
              help="The number of episodes to run at each epoch. Takes the "
              "priority over min_epoch_steps if set.")
@click.option("--episodic-update", default=False, show_default=True,
              type=click.BOOL,
              help="Whether to update the model after each episode such that "
              "one epoch is exaclty one episode.")
@click.option("--data-label", default="", show_default=True, type=click.STRING,
              help="Data label when evaluating different initial trace.")
@click.option("--model-env-module", default='model_env', show_default=True,
              type=click.STRING, help="Which model environment module to use. The "
              " default is to use the model_env module based on pandas. For faster "
              " computations use the numpy_model_env one.")
@click.option("--num-envs", default=10, show_default=True, type=click.INT,
              help="The number of environments to consider for the sb3 compatible "
              "vectorized model environment sb3_model_vec_env.")
@click.option("--seed", default=99999, show_default=True,
              help="Seed of the random number generator. Only the numpy and "
              "pytorch global random generators are seeded.")
@click.option("--partial-fit", default=False, show_default=True,
              help="If we want to pass the model from the previous epoch.")
@click.option("--epoch-resume", default=None, show_default=True,
              type=click.INT,
              help="If we want to resume training, the epoch from where "
              "we resume training. The training will resume from the agent if there is "
              "one saved, otherwise from the model is there is one saved (note that in "
              "this case the agent will be trained from scratch, othwersise from the "
              "past traces (note that in this case both the model and the agent will be"
              " trained from scratch). The first epoch to be run will be epoch_resume "
              "+ 1.")
@click.option("--save-model", default=True, show_default=True,
              help="Whether to save the trained_model.pkl at each epoch.")
@click.option("--save-agent", default=True, show_default=True,
              help="Whether to save the trained agent at each epoch.")
@click.option("--num-samples", default=1, show_default=True,
              help="Number of samples to use for ray-tune, in particular when"
              "doing a random search")
@click.option("--mar-threshold", default=None, show_default=True,
              type=click.FLOAT,
              help="The number of epochs from which we suppose convergence."
              "If None, read from metadata. If no metadata, set to 0.5")
def mbrl_hyperopt_command(agent_name, submission,
                          n_epochs, min_epoch_steps, min_random_steps,
                          n_epoch_episodes,
                          episodic_update,
                          data_label,
                          model_env_module, num_envs,
                          seed, partial_fit,
                          epoch_resume,
                          save_model, save_agent,
                          num_samples, mar_threshold):
    return mbrl_hyperopt(
        agent_name, submission,
        n_epochs, min_epoch_steps, min_random_steps, n_epoch_episodes,
        episodic_update,
        data_label,
        model_env_module, num_envs,
        seed, partial_fit,
        epoch_resume,
        save_model, save_agent,
        num_samples, mar_threshold
    )


if __name__ == "__main__":
    mbrl_hyperopt_command()
