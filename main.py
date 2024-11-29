import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import params
import dataset
from model import MLP, LSTM, DELSTM, weight_init
import os
import json
import time

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
train_set = dataset.DatasetGenerator(train_data, arg.window_size, arg.forecast_length, arg.time_step, arg.dim_env)
val_set = dataset.DatasetGenerator(val_data, arg.window_size, arg.forecast_length, arg.time_step, arg.dim_env)
test_set = dataset.DatasetGenerator(test_data, arg.window_size, arg.forecast_length, arg.time_step, arg.dim_env)
train_loader = DataLoader(train_set, batch_size=arg.batch_size, shuffle=False)
# no minibatch in the val_set and test_set
val_loader = DataLoader(val_set, params.val_size - arg.window_size, shuffle=False)
test_loader = DataLoader(test_set, params.test_size - arg.window_size, shuffle=False)

# stepup model
if arg.model == 'MLP':
    model = MLP(32, dim_hidden=512, drop_rate=0.3, dim_output=24,
                feature='fused_feature')
    model.apply(weight_init)
if arg.model == 'LSTM':
    model = LSTM(dim_input=32, dim_hidden=128, num_layers=1, dim_output=24,
                 feature='fused_feature')
    model.apply(weight_init)
if arg.model == 'DELSTM':
    model = DELSTM(dim_input1=24, dim_input2=8, dim_hidden1=arg.hd1, dim_hidden2=arg.hd2, dim_feature=arg.fd,
                   dim_output=24, dim_env=8, num_layers1=1, num_layers2=1,
                   use_dual_encoder=arg.single_encoder,
                   use_attention=arg.no_attention,
                   given_env=arg.no_env)
    model.apply(weight_init)


def train():
    # use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model.to(device)

    # define optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate[arg.model], weight_decay=params.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # params for choosing the best model with the best val_mre in the val_set
    best_val_mre = 1e3
    result_test_me = 0
    result_test_mre = 0
    result_test_r2 = 0
    best_id = 0
    total_time = 0
    for epoch in range(params.num_epoch):
        # training
        start_time = time.time()
        model.train()
        train_loss = 0
        for batch_id, (train_inputs, train_labs, train_env) in enumerate(train_loader):
            train_inputs, train_labs, train_env = train_inputs.to(device), train_labs.to(device), train_env.to(device)
            # propagation
            if arg.model == 'MLP' or arg.model == 'LSTM':
                outputs = model(train_inputs)
            if arg.model == 'DELSTM':
                outputs = model(train_inputs, train_env)
            loss = criterion(outputs, train_labs)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        end_time = time.time()
        training_time = end_time - start_time
        total_time += training_time
        train_loss /= len(train_loader)
        # validation
        model.eval()
        val_loss = 0
        val_me = 0
        val_mre = 0
        test_loss = 0
        test_me = 0
        test_mre = 0
        test_r2 = 0
        with torch.no_grad():
            for val_inputs, val_labs, val_env in val_loader:
                val_inputs, val_labs, val_env = val_inputs.to(device), val_labs.to(device), val_env.to(device)
                # propagation
                if arg.model == 'MLP' or arg.model == 'LSTM':
                    outputs = model(val_inputs)
                    loss = criterion(outputs, val_labs)
                    preds = outputs.detach().cpu().numpy()
                    preds = dataset.inverse_scaler(preds, scaler, arg.forecast_length, dim_env=arg.dim_env)
                if arg.model == 'DELSTM':
                    outputs = model(val_inputs, val_env)
                    loss = criterion(outputs, val_labs)
                    preds = outputs.detach().cpu().numpy()
                    preds = dataset.inverse_scaler(preds, scaler, arg.forecast_length, dim_env=arg.dim_env)
                labs = val_labs.detach().cpu().numpy()
                labs = dataset.inverse_scaler(labs, scaler, arg.forecast_length, dim_env=arg.dim_env)
                mean_err = np.mean(np.abs(preds - labs))
                mean_relative_err = np.mean(np.abs(preds - labs) / labs)
                val_loss += loss.item()
                val_me += mean_err
                val_mre += mean_relative_err
            for test_inputs, test_labs, test_env in test_loader:
                test_inputs, test_labs, test_env = test_inputs.to(device), test_labs.to(device), test_env.to(device)
                # propagation
                if arg.model == 'MLP' or arg.model == 'LSTM':
                    outputs = model(test_inputs)
                    loss = criterion(outputs, test_labs)
                    preds = outputs.detach().cpu().numpy()
                    preds = dataset.inverse_scaler(preds, scaler, arg.forecast_length, dim_env=arg.dim_env)
                if arg.model == 'DELSTM':
                    outputs = model(test_inputs, test_env)
                    loss = criterion(outputs, test_labs)
                    preds = outputs.detach().cpu().numpy()
                    preds = dataset.inverse_scaler(preds, scaler, arg.forecast_length, dim_env=arg.dim_env)
                labs = test_labs.detach().cpu().numpy()
                labs = dataset.inverse_scaler(labs, scaler, arg.forecast_length, dim_env=arg.dim_env)
                mean_err = np.mean(np.abs(preds - labs))
                mean_relative_err = np.mean(np.abs(preds - labs) / labs)
                r2 = 1 - np.sum(np.square(preds - labs)) / np.sum(np.square(preds - np.mean(labs, axis=0)))
                test_loss += loss.item()
                test_me += mean_err
                test_mre += mean_relative_err
                test_r2 += r2
        val_loss /= len(val_loader)
        val_me /= len(val_loader)
        val_mre /= len(val_loader)
        test_loss /= len(test_loader)
        test_me /= len(test_loader)
        test_mre /= len(test_loader)
        test_r2 /= len(test_loader)
        print(
            f'epoch{epoch}, training time {training_time}, train loss {train_loss:.4f}, val loss {val_loss:.4f}, val_me {val_me:.4f}, val_mre {val_mre:.4f}, test loss {test_loss:.4f}, test_me {test_me:.4f}, test_mre {test_mre:.4f}, test_r2 {test_r2:.4f}')
        if val_mre < best_val_mre:
            torch.save(model.state_dict(), f'{arg.model}.pth')
            best_val_mre = val_mre
            best_id = epoch
            result_test_me = test_me
            result_test_mre = test_mre
            result_test_r2 = test_r2
    print(
        f'total training time {total_time}, obtain the best model at epoch{best_id},best val mre {best_val_mre:.4f}, test me {result_test_me:.4f}, test mre {result_test_mre:.4f}, test r2 {result_test_r2:.4f}')
    # print(f'the total training time is {total_time}')


train()


def test():
    model_path = os.path.join('.\pretrained', f'{arg.model}.pth')
    pred_path = os.path.join('.\data', f'{arg.model}.json')
    pred_loads = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labs, test_env in train_loader:
            labels = dataset.inverse_scaler(test_labs, scaler, arg.forecast_length, dim_env=arg.dim_env).tolist()
            if arg.model == 'MLP' or arg.model == 'LSTM':
                outputs = model(test_inputs)
                preds = dataset.inverse_scaler(outputs, scaler, arg.forecast_length, dim_env=arg.dim_env).tolist()
            if arg.model == 'DELSTM':
                outputs = model(test_inputs, test_env)
                preds = dataset.inverse_scaler(outputs, scaler, arg.forecast_length, dim_env=arg.dim_env).tolist()
            for pred in preds:
                pred_loads.append(pred)
    with open(pred_path, 'w') as f:
        json.dump(pred_loads, f, indent=4)


# test()
