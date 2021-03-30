import os
import click

import cloudpickle

import numpy as np

from rampwf.utils import assert_read_problem
from rampwf.utils.importing import import_module_from_source

from mbrltools.model_env import make_model_env_class


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--data-label', default='random', show_default=True,
              help='The data label.')
@click.option('--submission', default='starting_kit', show_default=True,
              help='The kit to score.')
@click.option('--n-traces', default=100, show_default=True,
              help='The number of traces generated at each instance.')
@click.option('--n-samples', default=100, show_default=True,
              help='The number of samples from which we generate traces.')
@click.option('--future-length', default=10, show_default=True,
              help='The horizon (number of generated steps).')
@click.option('--trace-suffix', default='', show_default=True,
              help='Suffix that we insert in model_dir/traces{}.npz.')
@click.option('--train', is_flag=True, show_default=True,
              help='Generate traces using the training set.')
@click.option('--mean-traces', default=False, is_flag=True, show_default=True,
              help='Average the <n-traces> traces into a single mean trace.')
@click.option('--start-fold', default=-1, show_default=True,
              help='The starting fold index. Default is fist fold.')
@click.option('--end-fold', default=-1, show_default=True,
              help='The ending fold index. Default is last fold.')
@click.option('--seed', default=None, show_default=True,
              help='Seed of the RNG used to draw the samples from which to '
              'generate the traces.')
def simulate_command(data_label, submission, n_traces, n_samples,
                     future_length, trace_suffix, train, mean_traces,
                     start_fold, end_fold, seed):
    """Simulate random traces from learned generative models.

    Run this from the benchmarks/<env> folder. Traces are saved in
    submissions/<submission>/training_output/
    <data-label>/fold_<i>/traces<trace-suffix>.npz. The shape is
    (<n-samples>, <n-traces>, <future-length>, n_dim) where n_dim
    is the number of system observables. The saved traces can be used in
    evaluation scripts (that compare them to ground truth) and in visualization
    scripts.

    We sample without replacement from all eligible points: we exclude the
    first problem._n_burn_in and last <future-length> points, so we can compare
    all generated observables to existing ground truth. If we want to use
    all eligible points, e.g., the test set has length 20000 with 40 traces of
    length 500 each, with n_burn_in=0 and <future-length> = 20, the number of
    excluded point is 800, so we set <n-samples> to 19200. When <train> is set,
    we also exclude the validation points obtained from problem.get_cv.
    When <mean-traces> is set, we take the mean over the <n-traces> dimension,
    but reshape the traces to (<n-samples>, 1, <future-length>, n_dim) so it
    can be fed into the evaluation scripts (that compare them to ground truth).
    """
    np.random.seed(seed)
    is_train = train
    if is_train:
        trace_suffix = '_train' + trace_suffix
    is_mean_traces = mean_traces
    problem = assert_read_problem()
    restart_name = problem._restart_name
    n_burn_in = problem._n_burn_in
    if n_burn_in >= 1:
        raise ValueError('Vectorized simulation with n_burn_in greater than or'
                         ' equal to 1 is not supported')
    metadata = problem.metadata
    observation_names = metadata['observation']
    action_names = metadata['action']
    restart_name = metadata['restart_name']
    n_dim = len(observation_names)

    env = import_module_from_source('env.py', 'env')
    Env = env.Env
    ModelEnv = make_model_env_class(Env)
    reward_module = import_module_from_source(
        'reward_function.py', 'reward_function')
    reward_func = reward_module.reward_func
    model_env = ModelEnv(None, problem, reward_func, metadata, None, seed=None)

    X_test, _ = problem.get_test_data(data_label=data_label)
    X_train, y_train = problem.get_train_data(data_label=data_label)
    folds = list(problem.get_cv(X_train, y_train))
    if start_fold == -1:
        start_fold = 0
    if end_fold == -1:
        end_fold = len(folds)
    submission_dir = os.path.join('submissions', submission)

    for fold_i, fold in enumerate(folds[start_fold:end_fold]):
        print(f'fold #{fold_i + start_fold}')
        model_dir = os.path.join(
            submission_dir, 'training_output', data_label,
            f'fold_{fold_i + start_fold}')
        try:
            with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
                trained_workflow = cloudpickle.load(f)
        except FileNotFoundError:
            train_is, _ = fold
            trained_workflow = problem.workflow.train_submission(
                submission_dir, X_train, y_train, train_is=train_is)
        model_env.model = trained_workflow

        if is_train:
            train_is, _ = fold
            X = X_train.iloc[train_is]
        else:
            X = X_test
        restart_is = np.r_[
            np.where(X[restart_name].to_numpy())[0], len(X)]
        # compute valid starting indices (based on the need of burn in
        # samples and a future length that does not go beyond the end of the
        # data).
        valid_start_is = np.concatenate(
            [np.arange(i + n_burn_in, i_plus_one - future_length)
             for i, i_plus_one in zip(restart_is[:-1], restart_is[1:])])
        # take samples at random as starting points in the set
        # of valid starts
        start_is = np.sort(
            np.random.choice(valid_start_is, n_samples, replace=False))

        # duplicate starting observations n_traces times
        observations_col_ind = X.columns.get_indexer(observation_names)
        start_observations = X.iloc[start_is, observations_col_ind]
        start_observations = start_observations.to_numpy()
        observation_rep = np.repeat(start_observations, n_traces, axis=0)
        restart_col_ind = X.columns.get_indexer([restart_name])
        start_restarts = X.iloc[start_is, restart_col_ind].to_numpy()
        restart_rep = np.repeat(start_restarts, n_traces, axis=0)

        model_env.add_observations_to_history(observation_rep, restart_rep)

        # build corresponding action sequences
        n_action_features = len(action_names)
        actions_col_ind = X.columns.get_indexer(action_names)
        actions_seq_ind = np.hstack(
            [start_i + np.arange(future_length) for start_i in start_is])
        action_sequences = X.iloc[
            actions_seq_ind, actions_col_ind].to_numpy()
        action_sequences = action_sequences.reshape(n_samples, future_length)
        action_sequences_rep = np.repeat(action_sequences, n_traces, axis=0)

        mc_observable_traces = np.zeros(
            (n_samples, n_traces, future_length, n_dim))
        for horizon in range(future_length):
            actions = action_sequences_rep[:, horizon].reshape(
                -1, n_action_features)
            observations, _, _, _ = model_env.step(actions)
            observations = observations.reshape(n_samples, n_traces, -1)
            mc_observable_traces[:, :, horizon, :] = observations

        if is_mean_traces:
            mc_observable_traces = mc_observable_traces.mean(axis=1)
            mc_observable_traces = mc_observable_traces.reshape(
                n_samples, 1, future_length, n_dim)

        traces_path = os.path.join(model_dir, f'traces{trace_suffix}.npz')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print(traces_path)
        np.savez(traces_path, start_is, mc_observable_traces)


if __name__ == "__main__":
    simulate_command()
