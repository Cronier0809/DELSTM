import torch
import torch.nn as nn
import torch.nn.functional as F
from params import arg


class MLP(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, drop_rate, feature='fused_feature'):
        super().__init__()
        self.fc1 = nn.Linear(arg.window_size * dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, arg.forecast_length * dim_output)
        self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()
        self.feature = feature

    def forward(self, x):  # input shape: [batch, dim_input(window_size * dim_feature]]
        if self.feature == 'load_feature':  # input dim window_size * 24
            x = x[:, :, -24:]
        if self.feature == 'env_feature':  # input dim window_size * 8
            x = x[:, :, :8]
        if self.feature == 'fused_feature':  # input dim window_size * 32
            pass
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        # shape: [batch, dim_input] → [batch, dim_hidden]
        z = self.relu(self.fc1(x))
        z = self.dropout(z)
        # shape: → [batch, dim_hidden] → [batch, dim_output]
        out = self.fc2(z)
        return out


class LSTM_Module(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(dim_hidden * arg.forecast_length, dim_output * arg.forecast_length)

    def forward(self, x):  # input shape: [batch, window_size, dim_feature]
        # shape: [batch, window_size, dim_feature] → [batch, window_size, dim_hidden]
        z_lstm, _ = self.lstm(x)
        # shape: [batch, window_size, dim_hidden] → [batch, forecast_length, dim_hidden]
        z_lstm = z_lstm[:, -arg.forecast_length:, :]
        # shape: [batch, forecast_length, dim_hidden] → [batch, dim_hidden * forecast_length]
        z_lstm = z_lstm.reshape(z_lstm.shape[0], -1)
        # shape: [batch, dim_hidden * forecast_length] → [batch, dim_out * forecast_length]
        out = self.fc(z_lstm)
        return out


class LSTM(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers=1, feature='fused_feature'):
        super().__init__()
        self.lstm = LSTM_Module(dim_input, dim_hidden, dim_output, num_layers)
        self.feature = feature

    def forward(self, x):  # input shape: [batch, window_size, dim_feature]
        if self.feature == 'load_feature':  # input dim 24
            x = x[:, :, -24:]
        if self.feature == 'env_feature':  # input dim 8
            x = x[:, :, :8]
        if self.feature == 'fused_feature':  # input dim 32
            pass
        out = self.lstm(x)
        return out


class Attention(nn.Module):
    def __init__(self, dim_feature):
        super().__init__()
        self.dim_feature = dim_feature
        self.weight = nn.Parameter(torch.randn(dim_feature))

    def forward(self, load_feature, environment_feature):  # shape: [batch, dim_feature]
        if load_feature.shape != environment_feature.shape:
            print('the numbers of load_feature and environment feature must be equal!')
        weight = self.weight.unsqueeze(1).repeat(1, self.dim_feature)   # shape [dim_feature, dim_feature]
        attention_scores = torch.matmul(load_feature, weight) + torch.matmul(environment_feature, weight)   # shape [batch, dim_feature]
        attention_weights = F.softmax(attention_scores, dim=1)  # shape [batch, dim_feature]
        fused_feature = attention_weights * load_feature + (1 - attention_weights) * environment_feature
        return fused_feature


class DELSTM(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_hidden1, dim_hidden2, dim_feature, dim_output, dim_env=8,
                 num_layers1=1, num_layers2=1, use_dual_encoder=True, use_attention=True, given_env=True):
        super().__init__()
        self.load_encoder = LSTM_Module(dim_input1, dim_hidden1, dim_feature, num_layers=num_layers1)
        self.environment_encoder = LSTM_Module(dim_input2, dim_hidden2, dim_feature, num_layers=num_layers2)
        self.attention = Attention(dim_feature * arg.forecast_length)
        self.fc_env = nn.Linear(dim_env * arg.forecast_length, dim_feature * arg.forecast_length)
        self.fc = nn.Linear(dim_feature * arg.forecast_length, dim_output * arg.forecast_length)
        self.relu = nn.ReLU()
        self.use_dual_encoder = use_dual_encoder
        self.use_attention = use_attention
        self.given_env = given_env

    def forward(self, x, x_env):  # env shape [batch, forecast_length * 8]
        # x1 shape [batch, window_size, 24]         x2 shape [batch, window_size, 8]
        x1, x2 = x[:, :, 8:], x[:, :, :8]
        if self.use_dual_encoder:
            load_features = self.relu(self.load_encoder(x1))
            environment_features = self.relu(self.environment_encoder(x2))
            if self.given_env:
                env = self.relu(self.fc_env(x_env))
                environment_features = environment_features * 0.5 + env * 0.5
            else:
                environment_features = environment_features
            if self.use_attention:
                fused_features = self.attention(load_features, environment_features)
            else:
                fused_features = load_features * 0.5 + environment_features * 0.5
            out = self.fc(fused_features)

        else:
            # input dim = 32
            feature = self.relu(self.load_encoder(x))
            if self.given_env:
                env = self.relu(self.fc_env(x_env))
                feature = feature * 0.5 + env * 0.5
            out = self.fc(feature)
        return out


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)



