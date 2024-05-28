from tinynet import Tensor, draw_dot
from tinynet.models.mlp import MLP

import numpy as np
from optimx import make

model_name = f"new_model"
VERSION = "0.0"
MODEL_ENV = "dev"
model_db2 = make(
    f"cache/{model_name}-v{VERSION}",
    db_name="dqnmodel_test22.db",
    env=MODEL_ENV,
    db_type="diskcache",
)

data = [
    [2.0, 2.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

target = [
    [1.0, 0],
    [-1.0, 0],
    [-1.0, 0],
    [1.0, 0],
]  # true outputs - a binary classifier
# target = [1.0, -1.0, -1.0, 1.0] # true outputs - a binary classifier
x = data  # Tensor(data,requires_grad=True)
y = target  # Tensor(target,requires_grad=True)
# x.shape,y.shape


model = MLP(
    n_in=3,
    hidden_dims=[32, 16],
    n_out=2,
    activ="tanh",
    last_layer="iden",
    model_db=model_db2,
    eps=0.1,
    loss="MSE",
)

import time

epochs = 400
for epoch in range(epochs):
    y_pred = []
    losses = []
    s = time.monotonic()
    for i in range(4):
        model.learn(x[i], y[i], "mse")
    e = time.monotonic()
    t = e - s
    # loss_epoch = sum(losses)/ len(losses)
    # r2 = r2_score(y, y_pred)
    # r2=0.9
    # if epoch % 100 == 0 or epoch == epochs-1:
    #    print(f"epoch: {epoch} | loss: {loss_epoch:.2f} | R2: {r2:.2f} | time: {t:.2f} sec.")

loss = model.learn(x[i], y[i], "mse", print_cost=True)

for i in range(4):
    yy = model.predict(x[i], "mse")
    print(yy)

draw_dot(loss)
