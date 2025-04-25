import torch as t

class Attention:
    def __init__(self, d_model, d_head):
        self.d_model = d_model
        self.d_head = d_head
        self.Q = t.nn.Linear(d_model, d_head)
        self.K = t.nn.Linear(d_model, d_head)
        self.V = t.nn.Linear(d_model, d_head)

    def forward(self, input: t.Tensor["batch seq_len d_model"]):
        pass
        