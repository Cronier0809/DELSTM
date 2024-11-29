import argparse

features = ['date', 'is_holiday', 'is_vacation', 'is_weekend', 'temp_max', 'temp_min', 'temp_avg', 'humid_avg', 'rain']
# we only considerate the national holidays, e.g., Spring Festival, Mid-Autumn Festival, etc.
holidays = ['2016/1/1', '2016/1/2', '2016/1/3', '2016/2/4', '2016/2/5', '2016/2/6', '2016/2/7', '2016/2/8', '2016/2/9',
            '2016/2/10', '2016/2/11', '2016/2/12', '2016/2/13', '2016/2/14', '2016/2/22', '2016/4/2', '2016/4/3',
            '2016/4/4', '2016/4/30', '2016/5/1', '2016/5/2', '2016/6/9', '2016/6/10', '2016/6/11', '2016/9/15',
            '2016/9/16', '2016/9/17', '2016/10/1', '2016/10/2', '2016/10/3', '2016/10/4', '2016/10/5', '2016/10/6',
            '2016/10/7', '2016/12/31', '2017/1/1', '2017/1/2', '2017/1/23', '2017/1/24', '2017/1/25', '2017/1/26',
            '2017/1/27', '2017/1/28', '2017/1/29', '2017/1/30', '2017/1/31', '2017/2/1', '2017/2/2', '2017/4/2',
            '2017/4/3', '2017/4/4', '2017/4/29', '2017/4/30', '2017/5/1', '2017/5/2', '2017/5/3', '2017/5/28',
            '2017/5/29', '2017/5/30', '2017/10/1', '2017/10/2', '2017/10/3', '2017/10/4', '2017/10/5', '2017/10/6',
            '2017/10/7', '2017/10/8', '2017/12/30', '2017/12/31', '2018/1/1', '2018/2/11', '2018/2/12', '2018/2/13',
            '2018/2/14', '2018/2/15', '2018/2/16', '2018/2/17', '2018/2/18', '2018/2/19', '2018/2/20', '2018/2/21',
            '2018/4/5', '2018/4/6', '2018/4/7', '2018/4/29', '2018/4/30', '2018/5/1', '2018/6/16', '2018/6/17',
            '2018/6/18', '2018/9/22', '2018/9/23', '2018/9/24', '2018/10/1', '2018/10/2', '2018/10/3', '2018/10/4',
            '2018/10/5', '2018/10/6', '2018/10/7', '2018/12/30', '2018/12/31']
learning_rate = {'MLP': 1e-3, 'LSTM': 5e-3, 'DELSTM': 2e-2}
weight_decay = 0
num_test = 30
num_epoch = 500
err = 50
# for daily level input
train_size, val_size, test_size = 876, 110, 110
# for hourly level input
# train_size, val_size, test_size = 876 * 24, 110 * 24, 110 * 24

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='MLP', choices=['MLP', 'LSTM', 'DELSTM'])
parser.add_argument('-bs', '--batch_size', type=int, default=100)
parser.add_argument('-ts', '--time_step', type=int, default=1)
parser.add_argument('-ws', '--window_size', type=int, default=5)
parser.add_argument('-fl', '--forecast_length', type=int, default=1)
# daily level: dim_env=8    hourly level: dim_env=9
parser.add_argument('-denv', '--dim_env', type=int, default=8, choices=[8, 9])
parser.add_argument('-hd1', type=int, default=64, help='hidden feature dimension of load encoder')
parser.add_argument('-hd2', type=int, default=24, help='hidden feature dimension of environment encoder')
parser.add_argument('-fd', type=int, default=24, help='output feature dimension')
parser.add_argument('--single_encoder', action='store_false', help='without dual-encoder')
parser.add_argument('--no_attention', action='store_false', help='without attention mechanism')
parser.add_argument('--no_env', action='store_false', help='without given environment parameters')
parser.add_argument('--use_global_statistics', action='store_true',
                    help='use the global statistic metrics standardize the dataset')
# for ablation study of using the hour level data as input, you should adjust the model parameters manually
parser.add_argument('--hourly_level_input', action='store_true',
                    help='use the hour level data as input where the dim_input=10, dim_output=1, window_size=ws*24, forecast_length=fl*24')

arg = parser.parse_args()
