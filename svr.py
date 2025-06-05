import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import joblib

import params
import dataset

# add params
arg = params.arg

# load data
if arg.hourly_level_input:
    train_data, val_data, test_data, scaler = dataset.standardization(dim_env=arg.dim_env,
                                                                      feature_file='hourly_feature_data.json',
                                                                      use_global_statistics=arg.use_global_statistics)
else:
    train_data, val_data, test_data, scaler = dataset.standardization(dim_env=arg.dim_env,
                                                                      feature_file='feature_data.json',
                                                                      use_global_statistics=arg.use_global_statistics)

# data type: list
train_set = dataset.sample_sequence(train_data, arg.window_size, arg.forecast_length, arg.time_step, arg.dim_env)
val_set = dataset.sample_sequence(val_data, arg.window_size, arg.forecast_length, arg.time_step, arg.dim_env)
test_set = dataset.sample_sequence(test_data, arg.window_size, arg.forecast_length, arg.time_step, arg.dim_env)


def load_numpy_data(train_set, val_set, test_set, data_type='fused'):
    train_inputs, train_labs, train_env = train_set
    val_inputs, val_labs, val_env = val_set
    test_inputs, test_labs, test_env = test_set
    # data type: numpy array
    train_inputs, train_labs, train_env = np.array(train_inputs), np.array(train_labs), np.array(train_env)
    val_inputs, val_labs, val_env = np.array(val_inputs), np.array(val_labs), np.array(val_env)
    test_inputs, test_labs, test_env = np.array(test_inputs), np.array(test_labs), np.array(test_env)
    if data_type == 'load':
        x_train = train_inputs[:, :, :24].reshape(train_inputs.shape[0], -1)
        x_val = val_inputs[:, :, :24].reshape(val_inputs.shape[0], -1)
        x_test = test_inputs[:, :, :24].reshape(test_inputs.shape[0], -1)
    elif data_type == 'env':
        x_train = train_inputs[:, :, -8:].reshape(train_inputs.shape[0], -1)
        x_val = val_inputs[:, :, -8:].reshape(val_inputs.shape[0], -1)
        x_test = test_inputs[:, :, -8:].reshape(test_inputs.shape[0], -1)
    elif data_type == 'fused':
        x_train = train_inputs.reshape(train_inputs.shape[0], -1)
        x_val = val_inputs.reshape(val_inputs.shape[0], -1)
        x_test = test_inputs.reshape(test_inputs.shape[0], -1)
    else:
        raise ValueError(f"please select the correct data type: 'load', 'env' or 'fused'.")
    y_train = train_labs
    y_val = val_labs
    y_test = test_labs
    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = load_numpy_data(train_set, val_set, test_set, data_type='fused')

# hyperparameter space
param_grid = {
    'estimator__kernel': ['rbf', 'linear'],  # kernel function
    'estimator__C': [0.1, 0.5, 10],  # regularization
    'estimator__gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 3)),  # kernel coefficient
    'estimator__epsilon': [0.1]  # regression tolerance
}

base_svr = SVR()
model = MultiOutputRegressor(base_svr)

# generate dataset index (train set: -1     val set: 0)
test_fold = np.array([-1] * len(x_train) + [0] * len(x_val))
ps = PredefinedSplit(test_fold)
X = np.vstack([x_train, x_val])
Y = np.vstack([y_train, y_val])

# grid search for best parameters
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=ps,
    verbose=3,
    n_jobs=-1
)
grid_search.fit(X, Y)

# obtain the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f'best parameters: {best_params}')

# save model
joblib.dump(best_model, 'SVR.joblib')

# load model
pretrained_model = joblib.load('pretrained\SVR.joblib')
preds = pretrained_model.predict(x_test)

# calculate evaluation metrics
preds = dataset.inverse_scaler(preds, scaler, arg.forecast_length, dim_env=arg.dim_env)
labs = dataset.inverse_scaler(y_test, scaler, arg.forecast_length, dim_env=arg.dim_env)
mean_err = np.mean(np.abs(preds - labs))
mean_relative_err = np.mean(np.abs(preds - labs) / labs)
r2 = 1 - np.sum(np.square(preds - labs)) / np.sum(np.square(preds - np.mean(labs, axis=0)))
print(f'MAE: {mean_err}; MRE: {mean_relative_err}; R2: {r2}')