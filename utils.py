import pandas as pd
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm

import statsmodels.formula.api as smf
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


def prepare_missing_data(df, columns, groups_column, mu):
    """
    Add missing values to the dataset.
    :param df: dataset
    :param columns: columns to add missing values
    :param groups_column: column with groups
    :return: dataset with missing values
    """
    df_copy = df.copy()
    n_groups = df_copy[groups_column].nunique()
    c_probabilities = np.random.normal(0, 0.1, size=n_groups)
    c_probabilities += mu
    c_probabilities = np.clip(c_probabilities, 0, 1)

    # add missing values
    for i, column in enumerate(columns):
        for group in df_copy[groups_column].unique():
            df_copy.loc[df_copy[groups_column] == group, f"{column}_missing"] = df_copy.loc[df_copy[groups_column] == group, column].apply(lambda x: np.nan if np.random.uniform(0, 1) < c_probabilities[i] else x)
    return df_copy, c_probabilities

def split_train_test_by_env(df, groups_column, folds, test_size_n):
    '''
    split the data to train and test, each part contanin a different enviroment.
    :param: df - the dataset (type - data_frame)
    :param: groups_column - the column of the enviromets in the df.
    :param: folds - list of the enviroments
    :param: test_size_n - n from the environments they will be in the test
            (proportionally to the test size)
    '''
    envs_permutation = np.random.permutation(folds)
    test_folds = envs_permutation[:test_size_n].tolist()
    train_folds = envs_permutation[test_size_n:].tolist()

    # split data
    train_df = df[df[groups_column].isin(train_folds)].copy()
    test_df = df[df[groups_column].isin(test_folds)].copy()
    return train_df, test_df

def compute_mixed_effects_model(df, target_columns, other_columns, groups_column):
    """
    Compute mixed effects model.
    :param df: dataset
    :param target_column: target column
    :param groups_column: groups column
    :return: mixed effects model
    """
    models = {}
    # target_column_missing = target_column + '_missing'
    other_columns_formula = ' + '.join(other_columns)
    df_copy = df.copy()
    for target_column in target_columns:
        df_target = df_copy.drop(columns=[target_column]).copy()
        target_column_missing = target_column + '_missing'
        df_target = df_target.dropna(subset=[target_column_missing])
        models[target_column] = smf.mixedlm(f'{target_column} ~ {other_columns_formula}', df_copy, groups=df_copy[groups_column]).fit()
    return models

def compute_linear_regression_model(df, target_columns, other_columns, groups_column):
    """
    Compute multivariate regression model.
    :param df: dataset
    :param target_column: target column
    :param groups_column: groups column
    :return: mixed effects model
    """
    models = {}
    # target_column_missing = target_column + '_missing'
    other_columns_formula = ' + '.join(other_columns)
    df_copy = df.copy()
    for target_column in target_columns:
      df_target = df_copy.drop(columns=target_column).copy()
      target_column_missing = target_column + '_missing'
      df_target = df_target.dropna(subset=[target_column_missing])
      models[target_column] = smf.ols(f'{target_column_missing} ~ {other_columns_formula}',data=df_target).fit()
    return models

def compute_baseline_model(df, target_columns, other_columns):
    """
    Compute simple (mean fill) model.
    :param df: dataset
    :param target_column: target column
    :param groups_column: groups column
    :return: mixed effects model
    """
    models = {}
    # target_column_missing = target_column + '_missing'
    df_copy = df.copy()
    for target_column in target_columns:
        target_column_missing = target_column + '_missing'
        df_target = df_copy[other_columns+[target_column_missing]].dropna()
        models[target_column] = SimpleImputer(strategy='mean').fit(df_target)
    return models
    

def compute_KNN_model(df, target_columns, other_columns, n_neighbors=3):
    """
    Compute KNN model.
    :param df: dataset
    :param target_column: target column
    :param groups_column: groups column
    :return: mixed effects model
    """
    models = {}
    df_copy = df.copy()
    for target_column in target_columns:
        target_column_missing = target_column + '_missing'
        df_target = df_copy[other_columns+[target_column_missing]].dropna()
        models[target_column] = KNNImputer(n_neighbors=n_neighbors).fit(df_target)
    return models

model_types = {
    'mixed_effects': compute_mixed_effects_model,
    'linear_regression': compute_linear_regression_model,
    'KNN': compute_KNN_model,
    'baseline': compute_baseline_model
}

def impute_missing_values(df, target_column, other_columns, groups_column, test_size, split_by_env, model_type=['mixed_effects'], model_args={}, mu_limits=(0.1, 0.4), max_missing_rate=0.25, mus=None):
    """
    Impute missing values.
    :param df: dataset
    :param target_column: the target column
    :param other_columns: columns to use as features for predicting missing values
    :param groups_column: groups column (enviroment column)
    :param test_size: test size
    :param split_by_env: split by environment
    :param model_type: model type
    :param model_args: model arguments
    :param mu_limits: missing values limits
    :return: dataset with imputed missing values
    """

    if split_by_env:
        folds = df[groups_column].unique()
        test_size_n = max(int(test_size * len(folds)), 1)
        train_df, test_df = split_train_test_by_env(df, groups_column,
                                                    folds, test_size_n)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size)

    # get missing values rate for train, test
    if mus is None:
        mu_train, mu_test = np.random.uniform(mu_limits[0], mu_limits[1], size=2)
    else:
        mu_train, mu_test = mus
    # add missing values
    train_df, train_probs = prepare_missing_data(train_df,
                                    columns=target_column,
                                    groups_column=groups_column,
                                    mu=mu_train,
                                    max_missing_rate=max_missing_rate)
    test_df, test_probs = prepare_missing_data(test_df,
                                    columns=target_column,
                                    groups_column=groups_column,
                                    mu=mu_test,
                                    max_missing_rate=max_missing_rate)

    # complete missing values
    for mt in model_type:
        model_func_args = {'df': train_df, 'target_columns': target_column, 'other_columns': other_columns}
        if mt in ['mixed_effects', 'linear_regression']:
            model_func_args['groups_column'] = groups_column # add groups column
        elif mt == 'KNN':
            model_func_args['n_neighbors'] = model_args.get('n_neighbors', 3) # add n_neighbors
        model_func = model_types[mt]
        models = model_func(**model_func_args)
        for tc in models.keys():
            model = models[tc]
            if mt in ['baseline', 'KNN']:
                model_features = model.feature_names_in_
                train_df_trf = pd.DataFrame(model.transform(train_df[model_features]), columns=model_features)
                test_df_trf = pd.DataFrame(model.transform(test_df[model_features]), columns=model_features)
                
                train_df.loc[:, tc+f'_imp_{mt}'] = train_df_trf.loc[:, tc+'_missing']
                test_df.loc[:, tc+f'_imp_{mt}'] = test_df_trf.loc[:, tc+'_missing']
                
            else:
                train_df.loc[:, tc+f'_imp_{mt}'] = pd.Series(model.predict(train_df.drop(columns=[tc])))
                test_df.loc[:, tc+f'_imp_{mt}'] = pd.Series(model.predict(test_df.drop(columns=[tc])))
    return train_df, test_df, mu_train, mu_test


def bootstrapping(df, target_column, other_columns,
                                   groups_column, n_iterations=10,
                                   test_size=0.2, split_by_env=True,
                                   model_type='mixed_effects', model_args={}, mu_limits=(0.1, 0.4), max_missing_rate=0.25, controlled_mus=None):
    """
     Leave-one-out cross-validation.
    :param df: dataset
    :param target_column: the target column
    :param other_columns: columns to use as features for predicting missing values
    :param groups_column: groups column (enviroment column)
    :param n_iterations: number of iterations
    :param test_size: test size
    :param split_by_env: split by environment
    :param model_type: model type
    :param model_args: model arguments
    :param mu_limits: missing values limits
    :return: evaluation metrics
    """
    df_copy = df.copy()

    model_func_choice = []
    if model_type == 'all':
        # Note: keep the KNN to be the first model (sklearn too strict...)
        model_func_choice = ['KNN', 'mixed_effects', 'linear_regression', 'baseline']
    else:
        model_func_choice = [model_type]

    rmse_metric = lambda data, mt: np.sqrt(np.mean((data[target_column] - data[target_column+f'_imp_{mt}'])**2)) # RMSE = sqrt(mean((y_true - y_pred)^2))
    metrics = []
    train_probs, test_probs = [], []
    n_mus = 1
    if controlled_mus is not None:
        n_mus = controlled_mus
        mu_trains = np.full(n_mus, 0.1)
        mu_test_grid = np.linspace(0.1, 0.4, n_mus)
        mus = list(zip(mu_trains, mu_test_grid))
    
    pbar = tqdm(total=n_iterations * n_mus)
    for _ in tqdm(range(n_iterations)):
        for mu_iter in range(n_mus):
            if controlled_mus is None:
                mus_args = None
            else:
                mus_args = mus[mu_iter]
            train_df, test_df, train_p, test_p = impute_missing_values(df_copy, [target_column], other_columns, groups_column, test_size, split_by_env, model_func_choice, model_args, mu_limits, max_missing_rate, mus_args)
            # get indices of rows with missing values
            train_null_row_indices = train_df[train_df[target_column +'_missing'].isnull()].index
            test_null_row_indices = test_df[test_df[target_column +'_missing'].isnull()].index
            train_probs.append(train_p)
            test_probs.append(test_p)
            # compute evaluation metrics
            current_metrics = {}
            for mt in model_func_choice:
                # compute RMSE
                current_metrics[mt] = {
                    'train_rmse': rmse_metric(train_df, mt),
                    'test_rmse': rmse_metric(test_df, mt),
                    'train_rmse_null': rmse_metric(train_df.loc[train_null_row_indices], mt),
                    'test_rmse_null': rmse_metric(test_df.loc[test_null_row_indices], mt)
                }

            metrics.append(current_metrics)
            pbar.update(1)
    pbar.close()
    return metrics, train_probs, test_probs

def compute_mean_std_metrics(metrics):
    """
    Compute mean and std of metrics.
    :param metrics: evaluation metrics
    :return: mean and std of metrics
    """
    # calculate mean and std of metrics per model
    mean_metrics = {}
    std_metrics = {}
    for mt in metrics[0].keys():
        mean_metrics[mt] = defaultdict(float)
        std_metrics[mt] = defaultdict(float)
        for metric in metrics[0][mt].keys():
            mean_metrics[mt][metric] = np.nanmean([m[mt][metric] for m in metrics])
            std_metrics[mt][metric] = np.nanstd([m[mt][metric] for m in metrics])
    return mean_metrics, std_metrics

def error_plot(metrics, ax):
    """
    Plot error.
    :param metrics: evaluation metrics
    :param ax: plot axis
    """
    # convert metrics to flat dataframe
    flat_metrics = []
    for i, m in enumerate(metrics):
        for mt in m.keys():
            current_metrics = m[mt]
            current_metrics['iteration'] = i
            current_metrics['model'] = mt.replace('_', ' ').capitalize()
            flat_metrics.append(current_metrics)
    flat_metrics_df = pd.DataFrame(flat_metrics)
    flat_metrics_df = flat_metrics_df.melt(id_vars=['iteration', 'model'], value_vars=['train_rmse', 'test_rmse', 'train_rmse_null', 'test_rmse_null'], var_name='metric', value_name='value')
    # grouped boxplot of the error by split
    flat_metrics_df.reset_index(drop=True, inplace=True)
    flat_metrics_df['metric'] = flat_metrics_df['metric'].apply(lambda x: x.replace('_rmse', ''))
    # box plot of the error by split
    sns.boxplot(data=flat_metrics_df, x='metric', y="value", hue='model', ax=ax, showfliers=False, palette='pastel', width=0.5)

    ax.set_ylabel(r'$\sqrt{MSE}$', fontsize=14)
    ax.set_xlabel('Metric', fontsize=14)
    ax.legend(fontsize=14)

def mu_diff_vs_rmse_plot(metrics, train_probs_list, test_probs_list, ax):
    """
    Plot mu diff vs train, test rmse.
    :param metrics: evaluation metrics
    :param train_probs_list: list of train missing values probabilities
    :param test_probs_list: list of test missing values probabilities
    :param ax: plot axis
    """
    test_rmse_per_model = defaultdict(list)
    for m in metrics:
        for mt in m.keys():
            test_rmse_per_model[mt].append(m[mt]['test_rmse'])
    
    diff = [mu_train - mu_test for mu_train, mu_test in zip(train_probs_list, test_probs_list)]
    diff_dict = defaultdict(lambda: defaultdict(list))
    for mt in test_rmse_per_model.keys():
        for i, diff_val in enumerate(diff):
            diff_dict[mt][diff_val].append(test_rmse_per_model[mt][i])
    # get mean and std of test rmse per model
    mean_test_rmse_per_model = {mt: {diff_val: np.nanmean(rmse) for diff_val, rmse in diff_dict[mt].items()} for mt in diff_dict.keys()}
    std_test_rmse_per_model = {mt: {diff_val: np.nanstd(rmse) for diff_val, rmse in diff_dict[mt].items()} for mt in diff_dict.keys()}
    # plot mu diff vs train, test rmse
    for mt in test_rmse_per_model.keys():
        sorted_keys = sorted(mean_test_rmse_per_model[mt].keys())
        ax.errorbar(sorted_keys, [mean_test_rmse_per_model[mt][k] for k in sorted_keys], yerr=[std_test_rmse_per_model[mt][k] for k in sorted_keys], label=mt.replace('_', ' ').capitalize())
    ax.set_xlabel(r'$\mu_{train}-\mu_{test}$', fontsize=17)
    ax.set_ylabel(r'$\sqrt{MSE}$', fontsize=14)
    ax.legend(fontsize=14)