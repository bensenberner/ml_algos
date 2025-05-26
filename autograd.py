"""
a linear transformation and some relus
calculate the gradient wrt each parameter
apply it to some optimizer


y_pred = wx
dy/dw = x

loss = (y_true - y_pred)^2 = yt^2 - 2yt*yp + yp^2
d_loss/d_ypred = -2yt + 2yp
d_ypred/d_w1 = z
d_loss/d_w1 = dloss/d_ypred * d_ypred/d_w1
d_loss/d_w2 = dloss/d_ypred * d_ypred/d_z * d_z / d_w2
            = (-2yt + 2yp) * (w1) * ()
d_z / d_w2 = x

d_loss/dz = d_loss/d_ypred * d_ypred/d_z
d_ypred/d_z = w1

mylayer = Layer(w)

pred = mylayer(x)

def loss(pred, y_true):
    return (y_true - y_pred)^2


# grad = layer.calc_grad(expected_output)
# layer.backward(grad)
def loss_grad(y_true, y_pred):
    rerturn ...
LAYER1 = z = x * w2
LAYER2 = y_pred = z * w1
layer1 update = d_loss/d_w2

dloss_dz = dloss_ypred * w1
_, dloss_dz = layer2.calc_grads(dloss_ypred, input(opt)) # also updating d_w1 internally
_, dloss_dx = layer1.calc_grad(dloss_dz)
for layer in (layer1, layer2):
  layer.update_weights_with_grad()
"""
def loss_grad(y_true, y_pred):
    return -2 * y_true + 2 * y_pred
from jaxtyping import Float
import numpy as np

class Layer:
    def __init__(self, w: float):
        self.w = w
        self.input = None
        self.dw = 0
    
    # def forward(self, x: Float[np.array, "d_in"]) -> Float[np.array, "d_out"]:
    def forward(self, x: float) -> float:
        self.input = x
        return x * self.w

    def calc_grad_wrt_input_and_update_grad_wrt_weight(self, output_grad):
        update = output_grad * self.input
        self.dw = update
        return output_grad * self.w
    
l1 = Layer(w=4)
l2 = Layer(w=8)
x = 9
y_true = 12

for _ in range(10):
    z = l1.forward(x)
    y_pred = l2.forward(z)
    print(y_pred)
    d_loss_d_y = loss_grad(y_true, y_pred)
    dloss_dz = l2.calc_grad_wrt_input_and_update_grad_wrt_weight(d_loss_d_y)
    dloss_dx=l1.calc_grad_wrt_input_and_update_grad_wrt_weight(dloss_dz)

    for layer in (l1, l2):
        layer.w -= (layer.dw * 10e-6)
        layer.dw = 0

# l1 = Layer(w=4)
# l2 = Layer(w=8)
# x = 9
# z = l1.forward(x)
# y_pred = l2.forward(z)