# Multiple-Environment-Imputation

Testing of mixed-effects models to imput missing data in multiple-environment settings.

Each experiment is configured in the .yaml file, with the following parameters:
- `n_iterations`: number of iterations in the bootstrapping functions
- `test_size`: proportion of environments that should be allocated to the test set
- `split_by_env`: boolean flag indicating to split train/test by environment
- `model_type`: one of 5 options: [KNN, mixed_effects, linear_regression, baseline, all]. If choosing 'all' then tests all four methods.
- `model_args`: other arguments to pass to the imputing model (e.g n_neighbors for KNN)
- `mu_limits`: min and max for the mean values of the missing data rates.
- `controlled_mus`: If not None (i.e, its an integer), then tests all mu_test values between mu_train and 0.4. Otherwise, randomly draws mu_train, mu_test between mu_limits.
- `target_column`: the feature name(s) to test imputation on
- `other_columns`: the feature names to use for predicting `target_column`
- `groups_column`: the feature name indicating the data cluster IDs
- `path`: path to the dataset file
- `type`: type of dataset file, one of [csv, sav]
