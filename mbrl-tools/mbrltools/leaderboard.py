import pathlib

import numpy as np
import pandas as pd

from rampwf.utils import assert_read_problem

from .data_processing import get_metadata_dictionary, get_trace_df


def ci(std, n):
    """Standard 95% confidence interval."""

    return std / np.sqrt(n) * 1.645


def mar(trace_f_name):
    """Mean asymptotic reward (MAR)."""

    trace_df = pd.read_csv(trace_f_name)
    return trace_df['reward'].mean()


def minimum_mar(trace_f_name=None):
    """MAR of the random policy applied to the real system."""
    if trace_f_name is None:
        trace_f_name = pathlib.Path('data') / 'real_traces_random.csv'
    return mar(trace_f_name)


def optimum_mar(trace_f_name=None, agent='random_shooting'):
    """MAR of the agent applied to the real system.

    Agent is for example random shooting.
    Used as the optimum MAR when agent is used with models.
    """
    if trace_f_name is None:
        trace_f_name = pathlib.Path('data') /\
            'real_traces_{}.csv'.format(agent)
    return mar(trace_f_name)


def mrcp_score(seed_dirs, opt_mar=None, min_mar=None, threshold=None,
               n_smoothing=5, precision=-1, verbose=False):
    """Mean reward convergence pace and its 95% ci.

    See Section 2.2.2 in
    https://openreview.net/forum?id=p5uylG94S68
    How fast the submission/agent pair achieves
    threshold * the optimum RMAR.
    It needs to be called from the benchmarks/<env> folder.

    Parameters
    ----------
    seed_dirs : list of Path
        the folders where the epoch_* folders are
    opt_mar : float or None
        The optimum mar, typically achieved by running the
        agent on the real dynamics. If None, it's computed using
        optimum_mar(
            trace_f_name='data/real_traces_random_shooting.csv', agent)
    min_mar : float or None
        The minimum mar, typically achieved by running the
        random agent on the real dynamics. If None, it's computed using
        minimumum_mar()
    threshold : float or None
        the proportion of the optimal RMAR the algorithm needs to exceed
        if None, read from metadata
        if no metadata, set to 0.7
    n_smoothing : int
        the smoothing window size of the RMAR vs epoch id curve
    precision : int
        the precision of the returned score and its cofidence interval
    verbose : boolean
        if true, seed folders and seedwise scores printed
    Returns
    -------
    a tuple of (mrcp, 95% confidence interval)
    """
    try:
        metadata_path = pathlib.Path('data') / 'metadata.json'
        metadata = get_metadata_dictionary(metadata_path)
        if threshold is None:
            threshold = metadata['mrcp_threshold']
        reward_name = metadata['reward'][0]
    except FileNotFoundError:
        reward_name = 'reward'
        if threshold is None:
            threshold = 0.7

    if opt_mar is None:
        opt_mar = optimum_mar(agent='random_shooting')
    if min_mar is None:
        min_mar = minimum_mar()

    mrcps = np.empty(0)
    for seed_dir in seed_dirs:
        trace_df = get_trace_df(seed_dir, verbose=verbose)
        if trace_df is not None:
            rewards = trace_df.groupby('epoch_id').mean()[reward_name]
            epoch_lengths = trace_df.groupby('epoch_id').count()[reward_name]
            running_means = np.convolve(
                rewards, np.ones((n_smoothing,)) / n_smoothing, mode='same')
            rmars = (running_means - min_mar) / (opt_mar - min_mar)
            criteria = rmars > threshold
            good_epoch_ids = np.nonzero(criteria)[0]
            if len(good_epoch_ids) == 0:
                mrcp = np.inf
            else:
                first_good_epoch_id = good_epoch_ids[0]
                mrcp = epoch_lengths[:first_good_epoch_id].sum()
            if verbose:
                print('{}: {}'.format(seed_dir, mrcp))
            mrcps = np.append(mrcps, mrcp)
    mrcp_mean = round(mrcps.mean(), precision)
    mrcp_ci = round(ci(mrcps.std(), len(mrcps)), precision)
    return mrcp_mean, mrcp_ci


def individual_rmar(rewards, opt_mar, min_mar, precision):
    """Relative mean asymptotic reward and its 95% ci.

    See Section 2.2.2 in
    https://openreview.net/forum?id=p5uylG94S68

    Parameters
    ----------
    rewards : list-like
        array of rewards
    opt_mar : float
        The optimum mar, typically achieved by running the
        agent on the real dynamics.
    min_mar : float or None
        The minimum mar, typically achieved by running the
        random agent on the real dynamics. If None, it's computed using
        minimumum_mar()

    Returns
    -------
    a tuple of (rmar, 95% confidence interval)
    """
    rmar = (rewards.mean() - min_mar) / (opt_mar - min_mar)
    rmar_ci = ci(rewards.std(), len(rewards)) / (opt_mar - min_mar)
    return round(rmar, precision), round(rmar_ci, precision)


def individual_mar(rewards, precision):
    """Mean asymptotic reward and its 95% ci.

    See Section 2.2.2 in
    https://openreview.net/forum?id=p5uylG94S68

    Parameters
    ----------
    rewards : list-like
        array of rewards

    Returns
    -------
    a tuple of (rmar, 95% confidence interval)
    """
    mar = rewards.mean()
    mar_ci = ci(rewards.std(), len(rewards))
    return round(mar, precision), round(mar_ci, precision)


def mar_score(seed_dirs, threshold=None, precision=3, verbose=False):
    """Mean asymptotic reward and its 95% ci.

    See Section 2.2.2 in
    https://openreview.net/forum?id=p5uylG94S68
    Mean reward across last threshold * n_epochs and seeds.
    It needs to be called from the benchmarks/<env> folder.

    Parameters
    ----------
    seed_dirs : list of Path
        the folders where the epoch_* folders are
    threshold : int or None
        the number of epochs from which we suppose convergence
        if None, read from metadata
        if no metadata, set to 0.5
    precision : int
        the precision of the returned score and its cofidence interval
    verbose : boolean
        if true, seed folders and seedwise scores printed
    Returns
    -------
    a tuple of (rmar, 95% confidence interval)
    """
    try:
        metadata_path = pathlib.Path('data') / 'metadata.json'
        metadata = get_metadata_dictionary(metadata_path)
        if threshold is None:
            threshold = metadata['mar_threshold']
        reward_name = metadata['reward'][0]
    except FileNotFoundError:
        reward_name = 'reward'
        if threshold is None:
            threshold = 0.5

    mar_rewards = np.array([])
    for seed_dir in seed_dirs:
        trace_df = get_trace_df(seed_dir, verbose=verbose)
        if trace_df is not None:
            epoch_df = trace_df.groupby('epoch_id').mean().reset_index()
            rewards = epoch_df[
                epoch_df['epoch_id'] >= threshold][reward_name]
            mar_rewards = np.append(mar_rewards, rewards)
            if verbose:
                mar, mar_ci = individual_mar(rewards, precision)
                print('{}: {} +- {} on {} epochs'.format(
                    seed_dir, mar, mar_ci, len(rewards)))
    return individual_mar(mar_rewards, precision)


def rmar_score(seed_dirs, opt_mar=None, min_mar=None, threshold=None,
               precision=3, verbose=False):
    """Relative mean asymptotic reward and its 95% ci.

    See Section 2.2.2 in
    https://openreview.net/forum?id=p5uylG94S68
    Relative mean reward across last threshold * n_epochs and seeds.
    It needs to be called from the benchmarks/<env> folder.

    Parameters
    ----------
    seed_dirs : list of Path
        the folders where the epoch_* folders are
    opt_mar : float or None
        The optimum mar, typically achieved by running the
        agent on the real dynamics. If None, it's computed using
        optimum_mar(
            trace_f_name='data/real_traces_random_shooting.csv', agent)
    min_mar : float or None
        The minimum mar, typically achieved by running the
        random agent on the real dynamics. If None, it's computed using
        minimumum_mar()
    threshold : int or None
        the number of epochs from which we suppose convergence
        if None, read from metadata
        if no metadata, set to 0.5
    precision : int
        the precision of the returned score and its cofidence interval
    verbose : boolean
        if true, seed folders and seedwise scores printed
    Returns
    -------
    a tuple of (rmar, 95% confidence interval)
    """
    try:
        metadata_path = pathlib.Path('data') / 'metadata.json'
        metadata = get_metadata_dictionary(metadata_path)
        if threshold is None:
            threshold = metadata['mar_threshold']
        reward_name = metadata['reward'][0]
    except FileNotFoundError:
        reward_name = 'reward'
        if threshold is None:
            threshold = 0.5

    if opt_mar is None:
        opt_mar = optimum_mar(agent='random_shooting')
    if min_mar is None:
        min_mar = minimum_mar()

    rmar_rewards = np.array([])
    for seed_dir in seed_dirs:
        trace_df = get_trace_df(seed_dir, verbose=verbose)
        if trace_df is not None:
            epoch_df = trace_df.groupby('epoch_id').mean().reset_index()
            rewards = epoch_df[
                epoch_df['epoch_id'] >= threshold][reward_name]
            rmar_rewards = np.append(rmar_rewards, rewards)
            if verbose:
                rmar, rmar_ci = individual_rmar(
                    rewards, opt_mar, min_mar, precision)
                print('{}: {} +- {} on {} epochs'.format(
                    seed_dir, rmar, rmar_ci, len(rewards)))
    rmar_rewards = rmar_rewards[~np.isnan(rmar_rewards)]
    return individual_rmar(rmar_rewards, opt_mar, min_mar, precision)


def r2_stats_table(submission, data_label, submission_label=None,
                   traces_f_name=None, is_train=False):
    """R-squared, bias, and variance stats vs lookahead horizon.

    Requires to run mbrl-simulate to generate traces from a learned model.
    For example:
    ```
    mbrl-simulate --data-label linear --submission nn_det --n-traces 1\
    --n-samples 19200 --future-length 20 --trace-suffix _full
    ```
    future-length is the lookahead horizon, stats will be produced _up to_
    this horizon. The maximum meaningful n-samples is the size of the test
    set minus the number of episodes times the future length. If less, we
    sample time steps without replacement. trace-suffix _full will produce
    `traces_full.npz` in the training output folder, which is the default.
    Deterministic models need only n-traces 1.
    ```
    mbrl-simulate --data-label linear --submission darmdn --n-traces 5\
    --n-samples 3800 --future-length 20 --trace-suffix _full --mean-traces
    ```
    Stochastic models should simulate more than 1 trace, typically the same
    number as the number of particles in the dynamic run. mean-traces means
    that even though we generate 5 traces, we only save the mean. The R2 and
    variance stats are increased by the variance of the mean trace, so more
    traces will produce better stats.
    ```
    mbrl-simulate --data-label linear --submission nn_det --n-traces 1\
    --n-samples 4320 --future-length 20 --trace-suffix _full_train --train
    mbrl-simulate --data-label linear --submission darmdn --n-traces 5\
    --n-samples 564 --future-length 20 --trace-suffix _full_train --train\
    --mean-traces
    ```
    For training stats, set train to True and trace-suffix to _full_train.
    Both the simulator and this script take care of the folds (traces are
    generated from the train, ignoring validation episodes).

    Parameters
    ----------
    submission : str
        the name of the system model, submissions/<submission>
    data_label : str
        the data label, submissions/<submission>/training_output/<data_label>
    submission_label : str or None
        the name of the submission in the output table
        if None, identical to submission
    traces_f_name :  str or None
        the name of the trace file in
        submissions/<submission>/training_output/<data_label>/fold_<i>
        If None, 'traces_full_train.npz' if is_train is True and
        'traces_full.npz' otherwise
    is_train : Boolean
        whether to generate training or test curves
    Returns
    -------
    a pandas DataFrame with mean, std, lower and upper 95% confidence
    intervals for all output variables and all stats in {R-squared,
    squared bias, and variance}. Squared bias and variance are normalized
    by the sample variance as in the definition of R-squared. We also compute
    the mean stats across the output variables. These columns are called
    {r2,bias2,variance}_{mean,std,cil,ciu}. The first two columns are
    `lookahead` (horizon) from 1 to future-length (determined by the
    corresponding dimension in the trace file) and `method`
    (submission_label), so concatenating these data frames produces a direct
    input to altair plots with appropriate legends.
    """
    if submission_label is None:
        submission_label = submission
    if traces_f_name is None:
        if is_train:
            traces_f_name = 'traces_train.npz'
        else:
            traces_f_name = 'traces.npz'

    problem = assert_read_problem()
    X_train, y_train = problem.get_train_data(data_label=data_label)
    X_test, y_test = problem.get_test_data(data_label=data_label)
    folds = list(problem.get_cv(X_train, y_train))

    # mask = np.abs(np.arctan2(X_df['sin_theta_1'], X_df['cos_theta_1']))\
    #        < 0.1 * np.pi
    # mask = (X_df['sin_theta_1'] > X_df['sin_theta_1'].quantile(0.1)) &\
    #        (X_df['sin_theta_1'] < X_df['sin_theta_1'].quantile(0.9))
    # mask = np.ones(len(X_df), dtype='bool')
    n_folds = len(folds)
    future_length = 10
    output_names = problem._target_column_observation_names
    n_dim = len(output_names)
    stats_df = pd.DataFrame(
        1 + np.arange(future_length), columns=['lookahead'])
    stats_df['method'] = submission_label

    r2s = np.empty((n_dim, n_folds, future_length))
    bias2s = np.empty((n_dim, n_folds, future_length))
    variances = np.empty((n_dim, n_folds, future_length))
    stats = ['r2', 'bias2', 'variance']
    stat_containers = [r2s, bias2s, variances]

    for o_i, output_name in enumerate(output_names):
        for fold_i, fold in enumerate(folds):
            npzfile = np.load(
                pathlib.Path('submissions') / submission /
                'training_output' / data_label / f'fold_{fold_i}' /
                traces_f_name)
            if is_train:
                train_is, _ = fold
                X_df = X_train.iloc[train_is]
            else:
                X_df = X_test
            start_is = npzfile['arr_0']
            mc_observable_traces = npzfile['arr_1']
            for n_lookahead in range(future_length):
                traces = mc_observable_traces[:, :, n_lookahead, o_i]
                # traces = traces[mask[np.array(start_is) + n_lookahead + 1]]
                mean_predictions = traces.mean(axis=1)
                ground_truth = X_df[
                    output_name][np.array(start_is) + n_lookahead + 1].values
                # ground_truth_masked = ground_truth[
                #     mask[np.array(start_is) + n_lookahead + 1]]
                bias2 = (ground_truth - mean_predictions).mean() ** 2 /\
                    ground_truth.var()
                variance = (ground_truth - mean_predictions).var() /\
                    ground_truth.var()
                mse = bias2 + variance
                r2 = 1 - mse
                r2s[o_i, fold_i, n_lookahead] = r2
                bias2s[o_i, fold_i, n_lookahead] = bias2
                variances[o_i, fold_i, n_lookahead] = variance
        for stat, stat_container in zip(stats, stat_containers):
            mean = stat_container[o_i].mean(axis=0)
            std = stat_container[o_i].std(axis=0)
            ci_o = ci(std, n_folds)
            stats_df[f'{output_name}_{stat}_mean'] = mean
            stats_df[f'{output_name}_{stat}_std'] = std
            stats_df[f'{output_name}_{stat}_cil'] = mean - ci_o
            stats_df[f'{output_name}_{stat}_ciu'] = mean + ci_o

    for stat, stat_container in zip(stats, stat_containers):
        mean = np.array([stats_df[f'{o}_{stat}_mean'] for o in output_names])\
            .mean(axis=0)
        std = np.sqrt(np.array(
            [stats_df[f'{o}_{stat}_std'] ** 2 for o in output_names])
            .mean(axis=0))
        ci_o = ci(std, n_folds * n_dim)
        stats_df[f'{stat}_mean'] = mean
        stats_df[f'{stat}_std'] = std
        stats_df[f'{stat}_cil'] = mean - ci_o
        stats_df[f'{stat}_ciu'] = mean + ci_o
    return stats_df


def ks_stats_table(submission, data_label, submission_label=None,
                   traces_f_name='traces.npz', is_train=False):
    """Kolmogorov-Smirnov stats vs lookahead horizon.

    Untested for now
    """
    if submission_label is None:
        submission_label = submission

    problem = assert_read_problem()
    X_train, y_train = problem.get_train_data(data_label=data_label)
    X_test, y_test = problem.get_test_data(data_label=data_label)
    folds = list(problem.get_cv(X_train, y_train))

    # mask = np.abs(np.arctan2(X_df['sin_theta_1'], X_df['cos_theta_1']))\
    #        < 0.1 * np.pi
    # mask = (X_df['sin_theta_1'] > X_df['sin_theta_1'].quantile(0.1)) &\
    #        (X_df['sin_theta_1'] < X_df['sin_theta_1'].quantile(0.9))
    # mask = np.ones(len(X_df), dtype='bool')
    n_folds = len(folds)
    future_length = 20
    output_names = problem._target_column_observation_names
    n_dim = len(output_names)
    stats_df = pd.DataFrame(
        1 + np.arange(future_length), columns=['lookahead'])
    stats_df['method'] = submission_label

    kss = np.empty((n_dim, n_folds, future_length))
    stats = ['ks']
    stat_containers = [kss]

    for o_i, output_name in enumerate(output_names):
        for fold_i, fold in enumerate(folds):
            npzfile = np.load(
                pathlib.Path('submissions') / submission /
                'training_output' / data_label / f'fold_{fold_i}' /
                traces_f_name)
            if is_train:
                train_is, _ = fold
                X_df = X_train.iloc[train_is]
            else:
                X_df = X_test
            start_is = npzfile['arr_0']
            mc_observable_traces = npzfile['arr_1']
            for n_lookahead in range(future_length):
                traces = mc_observable_traces[:, :, n_lookahead, o_i]
                # traces = traces[mask[np.array(start_is) + n_lookahead + 1]]
                ground_truth = X_df[
                    output_name][np.array(start_is) + n_lookahead + 1].values
                # ground_truth_masked = ground_truth[
                #     mask[np.array(start_is) + n_lookahead + 1]]
                quantiles = np.sort(np.array(
                    [g > m for g, m in zip(ground_truth, traces)]).sum(
                    axis=1))
                kss[o_i, fold_i, n_lookahead] = np.max(np.abs(
                    quantiles - range(len(quantiles)))) / len(quantiles)
        for stat, stat_container in zip(stats, stat_containers):
            mean = stat_container[o_i].mean(axis=0)
            std = stat_container[o_i].std(axis=0)
            ci_o = ci(std, n_folds)
            stats_df[f'{output_name}_{stat}_mean'] = mean
            stats_df[f'{output_name}_{stat}_std'] = std
            stats_df[f'{output_name}_{stat}_cil'] = mean - ci_o
            stats_df[f'{output_name}_{stat}_ciu'] = mean + ci_o

    for stat, stat_container in zip(stats, stat_containers):
        mean = np.array([stats_df[f'{o}_{stat}_mean'] for o in output_names])\
            .mean(axis=0)
        std = np.sqrt(np.array(
            [stats_df[f'{o}_{stat}_std'] ** 2 for o in output_names])
            .mean(axis=0))
        ci_o = ci(std, n_folds * n_dim)
        stats_df[f'{stat}_mean'] = mean
        stats_df[f'{stat}_std'] = std
        stats_df[f'{stat}_cil'] = mean - ci_o
        stats_df[f'{stat}_ciu'] = mean + ci_o
    return stats_df


def load_submission_scores(submission_path, data_label=None,
                           metric=None, step=None, verbose=False):
    """Load a score data frame for a single submission.

    This is a pure RAMP function that should be ported into the RAMP
    library eventually.

    Parameters
    ----------
    submission : str
        the name of the submission, submissions/<submission>
    data_label : str
        the data label, submissions/<submission>/training_output/<data_label>
    metric : str or None
        the name of the metrics in the output table
        if None, take all metrics
    step : str or None
        the name of the steps in the output table
        If None, step = ['train', 'valid', 'test']
    is_train : Boolean
        whether to generate training or test curves
    Returns
    -------
    a pandas DataFrame with scores as columns, and index as [fold, step]
    """

    training_output_path = submission_path / 'training_output'
    if data_label is not None and data_label != '':
        training_output_path = training_output_path / data_label
    if not training_output_path.exists():
        if verbose:
            print(f'{training_output_path} does not exist')
        return None
    folds_path = training_output_path.glob('*fold_*')
    data = {}
    for fold_path in folds_path:
        score_path = fold_path / 'scores.csv'
        if not score_path.exists():
            if verbose:
                print(f'{score_path} does not exist')
        else:
            fold_id = int(str(fold_path).split('_')[-1])
            if verbose:
                print(f'adding fold {fold_id}: {score_path}')
            scores = pd.read_csv(score_path, index_col=0)
            scores.columns.name = 'score'
            data[fold_id] = scores
    df = pd.concat(data, names=['fold'])
    metric = metric if metric else slice(None)
    step = step if step else slice(None)
    return df.loc[(slice(None), step), metric]
