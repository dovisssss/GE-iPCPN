import torch
import torch.nn as nn
import torch.fft as fft
from setuptools.dist import sequence


class Network(nn.Module):
    def __init__(self, cfg, velocity_flag=False):
        super(Network, self).__init__()
        self.input_dim = cfg.model.input_dim
        self.output_dim = cfg.model.output_dim
        self.lifting_channels = cfg.model.lifting_dim
        self.filters = cfg.model.filters
        self.kernel_size = cfg.model.kernel_size
        if velocity_flag:
            self.modes = cfg.model.velocity_modes
        else:
            self.modes = cfg.model.modes
        self.fno_flag = cfg.model.fno_flag

        self.conv1d = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            padding='same'
        )#未指定激活函数，应为tanh

        self.lstm = nn.LSTM(
            input_size=self.filters,#源码为input_shape=(None, self.input_dim)
            hidden_size=128,
            batch_first=True
        )

        self.swish = nn.SiLU() #silu是beta=1的swish
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, 32)
        self.dense3 = nn.Linear(32, self.output_dim)

    def EncoderP(self,inputs):
        #CNN-LSTM model for EncoderP
        x = self.conv1d(inputs)
        x = self.lstm(x)
        x = self.swish(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x

    def SpectralConv1d(self,x, stage):
        """Second part of the FNO model: Defining F(x) --> U
           Args:
               x: input tensor [batch, sequence_length, num_channels]
               stage: stage of the model
           Returns:
               x: output tensor"""
        sequence_length = x.shape[1]
        xo_fft = fft.rfft



