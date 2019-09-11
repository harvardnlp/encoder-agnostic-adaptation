import torch
import torch.nn as nn
import math

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MLP(nn.Module):
    def __init__(self, n_embd, n_state, dropout):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)
        self.act = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.c_fc.weight.data.normal_(std=0.02)
        self.c_fc.bias.data.zero_()
        self.c_proj.weight.data.normal_(std=0.02)
        self.c_proj.bias.data.zero_()

    def forward(self, x):
        """
            x is input, [T, B, n_state]
        """
        h = self.dropout_1(self.act(self.c_fc(x)))
        h2 = self.dropout_2(self.c_proj(h))
        return h2
