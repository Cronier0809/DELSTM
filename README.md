## DELSTM
An efficient dual-encoder LSTM model

### Dataset discription
The raw data consists of three years of power load data from a southern city in China, covering the period from January 1, 2016, to December 31, 2018, is saved in `./data/raw_data.csv`.
We processed the raw data and saved the feature data with daily-level inputs in `./data/feature_data.json`, saved the feature data with hourly-level inputs in `./data/hourly_feature_data.json`.
### Run
We provide a unified interface in `main.py`.
We also provide the pretrained MLP, LSTM, and DELSTM models in file `./pretrained` (`MLP.pth`, `LSTM.pth`, `DELSTM.pth`) for one-day ahead forecasting, and the forecasting results are saved in file `./data` (`MLP_pred.json`, `LSTM_pred.json`, `DELSTM_pred.json`).

Here are some examples for running our project.
#### For daily-level inputs:
```
Example1: (train MLP\LSTM\DELSTM for one-day ahead forecasting)
-m MLP\LSTM\DELSTM -ws 5 -fl 1 -denv 8

Example2: (train MLP\LSTM\DELSTM for two-day ahead forecasting)
-m MLP\LSTM\DELSTM -ws 5 -fl 2 -denv 8
```

#### For hourly-level inputs:
Remember to adjust the sample size and input dimension manually.
```
Example1: (one-hour ahead forecasting)
--hourly_level_input -m LSTM -ws 5 -fl 1 -denv 9

Example2: (24-hour ahead forecasting)
--hourly_level_input -m LSTM -ws 5*24 -fl 1*24 -denv 9
```

#### For ablation study:
```
Hourly level input:
--hourly_level_input -m LSTM -ws 5*24 -fl 1*24 -denv 9

Use global statistics:
--use_global_statistics -m DELSTM -ws 5 -fl 1 -denv 8

w/o dual-encoder:
--single_encoder -m DELSTM -ws 5 -fl 1 -denv 8

w/o attention:
--no_attention -m DELSTM -ws 5 -fl 1 -denv 8

w/o given env:
--no_env -m DELSTM -ws 5 -fl 1 -denv 8

Dual-encoder only:
--no_attention --no_env -m DELSTM -ws 5 -fl 1 -denv 8
```



