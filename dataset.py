import os
import json
import copy
import torch
import params
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

data_path = '.\\data'


def json_data(sample_file, feature_file, hour_feature_file, raw_file='raw_data.csv', data_path=data_path,
              input_level='daily'):
    sample_path = os.path.join(data_path, sample_file)
    feature_path = os.path.join(data_path, feature_file)
    hour_feature_path = os.path.join(data_path, hour_feature_file)
    file_path = os.path.join(data_path, raw_file)
    raw_data = pd.read_csv(file_path)
    length = len(raw_data)
    samples = []
    features = []
    hour_features = []
    id = 0
    if input_level == 'daily':
        for row_index in range(0, length, 96):  # the raw data samples every 15 min, 96 sampling points within a day
            # get the daily level features
            feature = []
            line_data = raw_data.iloc[row_index].values.tolist()
            date, _ = line_data[0].split()
            year, month, day = date.split('/')
            week_id = datetime.strptime(date, "%Y/%m/%d").weekday()
            is_holiday = int(date in params.holidays)
            is_vacation = int(int(month) in [7, 8])
            is_weekend = int(week_id in [5, 6])
            temp_max, temp_min, temp_avg, humid_avg, rain = line_data[2], line_data[3], line_data[4], line_data[5], \
                line_data[6]
            feature.extend([is_holiday, is_vacation, is_weekend, temp_max, temp_min, temp_avg, humid_avg, rain])
            # get the daily level power loads (we only retain 24 sampling points)
            loads = []
            for time_index in range(0, 96, 4):
                time_data = raw_data.iloc[row_index + time_index].values.tolist()
                load = time_data[1]
                loads.append(load)
                feature.append(load)
            sample = {'id': id,
                      'date': date,
                      'is_holiday': is_holiday,
                      'is_vacation': is_vacation,
                      'is_weekend': is_weekend,
                      'temp_max': temp_max,
                      'temp_min': temp_min,
                      'temp_avg': temp_avg,
                      'humid_avg': humid_avg,
                      'rain': rain,
                      'loads': loads}
            samples.append(sample)
            features.append(feature)
            id += 1
        with open(sample_path, 'w') as sample_data:
            json.dump(samples, sample_data, indent=4)
        with open(feature_path, 'w') as feature_data:
            json.dump(features, feature_data, indent=4)
    if input_level == 'hourly':
        for row_index in range(0, length, 96):  # the raw data samples every 15 min, 96 sampling points within a day
            # get the hour level features
            feature = []
            line_data = raw_data.iloc[row_index].values.tolist()
            date, _ = line_data[0].split()
            year, month, day = date.split('/')
            week_id = datetime.strptime(date, "%Y/%m/%d").weekday()
            is_holiday = int(date in params.holidays)
            is_vacation = int(int(month) in [7, 8])
            is_weekend = int(week_id in [5, 6])
            temp_max, temp_min, temp_avg, humid_avg, rain = line_data[2], line_data[3], line_data[4], line_data[5], \
                line_data[6]
            feature.extend([is_holiday, is_vacation, is_weekend, temp_max, temp_min, temp_avg, humid_avg, rain])
            hour_id = 0
            for time_index in range(0, 96, 4):
                temp_feature = copy.deepcopy(feature)
                temp_feature.append(hour_id)
                time_data = raw_data.iloc[row_index + time_index].values.tolist()
                load = time_data[1]
                temp_feature.append(load)
                hour_features.append(temp_feature)
                hour_id += 1
        with open(hour_feature_path, 'w') as feature_data:
            json.dump(hour_features, feature_data, indent=4)
    print(f'Raw data has been processed and saved!')



def standardization(dim_env=8, feature_file='feature_data.json', data_path=data_path, use_global_statistics=False):
    feature_path = os.path.join(data_path, feature_file)
    if not os.path.exists(feature_path):
        json_data(sample_file='sample_data.json', feature_file='feature_data.json',
                  hour_feature_file='hourly_feature_data.json')
    with open(feature_path, 'r') as feature_data:
        features = np.array(json.load(feature_data))
    train_size, val_size, test_size = params.train_size, params.val_size, params.test_size
    train_features = features[:train_size]
    val_features = features[train_size:train_size + val_size]
    test_features = features[train_size + val_size:]
    # standardization (Max-Min), To prevent information leakage, only use the statistical metrics from the train_data.
    if use_global_statistics:
        scaler = MinMaxScaler()
        data_max = np.max(train_features, axis=0)
        data_min = np.min(train_features, axis=0)
        scale = 1 / (data_max - data_min)
        global_load_max = np.max(train_features[:, dim_env:])
        global_load_min = np.min(train_features[:, dim_env:])
        scale[dim_env:] = np.full(24, 1 / (global_load_max - global_load_min))
        data_max[dim_env:] = np.full(24, global_load_max)
        data_min[dim_env:] = np.full(24, global_load_min)
        scaler.data_max_ = data_max
        scaler.data_min_ = data_min
        scaler.data_range_ = data_max - data_min
        scaler.scale_ = scale
        scaler.min_ = -data_min * scaler.scale_
        train_data = scaler.transform(train_features)
        val_data = scaler.transform(val_features)
        test_data = scaler.transform(test_features)
    else:
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_features)
        val_data = scaler.transform(val_features)
        test_data = scaler.transform(test_features)

    return train_data, val_data, test_data, scaler



def inverse_scaler(data, scaler, forecast_length, dim_env):
    # data shape [batch, 24 * forcast_length]
    inverse_data = []
    for forecast in range(forecast_length):
        # for hourly level input
        if dim_env == 9:
            batch_data = data[:, forecast: (forecast + 1)]
        # for daily level input
        if dim_env == 8:
            batch_data = data[:, forecast * 24: (forecast + 1) * 24]
        min_values = scaler.data_min_[dim_env:]
        max_values = scaler.data_max_[dim_env:]
        scale = max_values - min_values
        inverse_batch_data = batch_data * scale + min_values
        inverse_data.append(inverse_batch_data)
    inverse_data = np.concatenate(inverse_data, axis=0)
    return inverse_data


def sample_sequence(data, window_size, forecast_length, time_step, dim_env):
    total_num = len(data)
    X = []
    Y = []
    X_env = []
    for seq_id in range(0, total_num, time_step):
        if seq_id + window_size + forecast_length > total_num:
            break
        x = data[seq_id:seq_id + window_size]
        X.append(np.array(x))
        y = np.array([label[dim_env:] for label in data[seq_id + window_size:seq_id + window_size + forecast_length]])
        y = y.reshape(-1)
        Y.append(y)
        x_env = np.array(
            [label[:dim_env] for label in data[seq_id + window_size:seq_id + window_size + forecast_length]])
        x_env = x_env.reshape(-1)
        X_env.append(x_env)
    samples = [X, Y, X_env]
    return samples



class DatasetGenerator(Dataset):
    def __init__(self, data, window_size, forecast_length, time_step, dim_env):
        self.X = sample_sequence(data, window_size, forecast_length, time_step, dim_env)[0]
        self.Y = sample_sequence(data, window_size, forecast_length, time_step, dim_env)[1]
        self.X_env = sample_sequence(data, window_size, forecast_length, time_step, dim_env)[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        x_item = torch.tensor(self.X[item], dtype=torch.float32)
        y_item = torch.tensor(self.Y[item], dtype=torch.float32)
        env_item = torch.tensor(self.X_env[item], dtype=torch.float32)
        return x_item, y_item, env_item

