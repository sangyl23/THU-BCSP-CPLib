import torch
import torch.nn as nn

class LSTMUnit(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers = 2):
        
        super(LSTMUnit, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)
        
    def forward(self, x, prev_hidden, prev_cell):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
        
        L, B, F = x.shape
        output = x.reshape(L * B, -1) 
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)       
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) 
        
        return output, cur_hidden, cur_cell

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.ifAD = configs.ifAD
        self.pred_len = configs.pred_len
        self.num_layers = configs.num_layers
        self.input_size = configs.input_size 
        self.hidden_size = configs.hidden_size
        self.features = configs.features
        self.model = LSTMUnit(configs.features, configs.input_size, configs.hidden_size, num_layers = configs.num_layers)
    
    def forecast(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, 
                 speed, time_interval, 
                 speed_norm, time_interval_norm,
                 t_idx_enc, t_idx_dec):
        
        x = x_enc_AD if self.ifAD else x_enc
        
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(x.device) 
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(x.device) 
        outputs = []
        for idx in range(seq_len + self.pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:,idx:idx+1,...].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output, prev_hidden, prev_cell) 
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).permute(1, 0, 2).contiguous()

        return outputs
    
    def forward(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, mask = None, 
                 speed = None, time_interval = None, 
                 speed_norm = None, time_interval_norm = None,
                 t_idx_enc = None, t_idx_dec = None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(
                x_enc=x_enc,
                x_enc_AD=x_enc_AD,
                x_mark_enc=x_mark_enc,
                x_dec=x_dec,
                x_dec_AD=x_dec_AD,
                x_mark_dec=x_mark_dec,
                speed=speed,
                time_interval=time_interval,
                speed_norm=speed_norm,
                time_interval_norm=time_interval_norm,
                t_idx_enc = t_idx_enc,
                t_idx_dec = t_idx_dec
            )
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None