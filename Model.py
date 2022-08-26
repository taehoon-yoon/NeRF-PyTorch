import torch
import torch.nn as nn


class NeRF(nn.Module):
    def __init__(self, depth=8, hidden_units=256, position_ch=3,
                 direction_ch=3, output_ch=4, skip_connection=[4], use_viewdirs=True):
        """
        input shape=(*, position_ch+direction_ch)
        inside the Model torch split(dim=-1)
        originally each position_ch, direction_ch should be position encoded

        output shape=(*,4) each correspond to ->
        output[...,0:3] will be color RGB and output[...,3] will be sigma

        :param depth: total Depth before Yellow Arrow, consult original Paper,default 8
        :param hidden_units: # of hidden unit, default 256
        :param position_ch: size of gamma(x), consult Original paper
        :param direction_ch: size of gamma(d), consult Original paper
        :param output_ch: only used when use_viewdirs is False, normally it will not be used
        :param skip_connection: place where skip connection will occur, default [4]
        :param use_viewdirs: Normally it is True, Set to False when gamma(d) is zero
        """

        super().__init__()
        self.position_ch = position_ch
        self.direction_ch = direction_ch
        self.skip_connection = skip_connection
        self.output_ch = output_ch
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList([nn.Linear(position_ch, hidden_units)] +
                                         [
                                             nn.Linear(hidden_units, hidden_units)
                                             if i not in skip_connection
                                             else nn.Linear(hidden_units + position_ch, hidden_units)
                                             for i in range(depth - 1)
                                         ])
        if self.use_viewdirs:
            self.sigma_layer = nn.Linear(hidden_units, 1)
            self.feature_linear = nn.Linear(hidden_units, hidden_units)
            self.view_linears = nn.Linear(hidden_units + direction_ch, hidden_units // 2)
            self.color_layer = nn.Linear(hidden_units // 2, 3)
        else:
            self.output_linear = nn.Linear(hidden_units, output_ch)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        input_pos, input_dir = torch.split(inputs, [self.position_ch, self.direction_ch], dim=-1)
        x = input_pos
        for idx, layer in enumerate(self.pts_linears):
            x = self.relu(layer(x))
            if idx in self.skip_connection:
                x = torch.cat([input_pos, x], dim=-1)

        if self.use_viewdirs:
            sigma = self.sigma_layer(x)
            x = self.feature_linear(x)  # yellow Arrow in Original paper
            x = torch.cat([x, input_dir], dim=-1)
            x = self.view_linears(x)
            x = self.relu(x)
            color = self.color_layer(x)
            outputs = torch.cat([color, sigma], dim=-1)
        else:
            outputs = self.output_linear(x)
        return outputs
