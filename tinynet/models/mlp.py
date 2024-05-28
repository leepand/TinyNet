from tinynet import Tensor, draw_dot
import tinynet.nn as nn
from tinynet.optim import SGD, Adam
from tinynet.state import (
    get_parameters,
    get_no_grad_params,
    load_state_dict,
    get_state_dict,
)
import copy


class Net:
    def __init__(
        self, n_in, n_out, hidden_dims=[32, 16], activ="tanh", last_layer="iden"
    ):
        active_int_type_mapping = {"tanh": "he", "sigmoid": "xavier", "relu": "he"}
        self.l1 = nn.Linear(
            n_in, hidden_dims[0], int_type=active_int_type_mapping[activ]
        )
        self.l2 = nn.Linear(
            hidden_dims[0], hidden_dims[1], int_type=active_int_type_mapping[activ]
        )
        self.l3 = nn.Linear(
            hidden_dims[1], n_out, int_type=active_int_type_mapping[activ]
        )
        self.activ = activ
        self.last_layer = last_layer

    def __call__(self, x):
        x = self.l1(x)
        if self.activ == "relu":
            x = x.relu()
            x = self.l2(x)
            x = x.relu()
        elif self.activ == "tanh":
            x = x.tanh()
            x = self.l2(x)
            x = x.tanh()
        elif self.activ == "sigmoid":
            x = x.sigmoid()
            x = self.l2(x)
            x = x.sigmoid()
        else:
            raise ValueError(f"activation {self.activ} is not implemented.")
        if self.last_layer == "iden":
            x = self.l3(x)
        elif self.last_layer == "sigmoid":
            x = self.l3(x)
            x = x.sigmoid()
        else:
            x = self.l3(x)
        return x


class MLP:
    def __init__(
        self,
        n_in,
        hidden_dims,
        n_out,
        activ="tanh",
        last_layer="iden",
        model_db=None,
        eps=0.1,
        loss="MSE",
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_dims = hidden_dims
        self.activ = activ
        self.last_layer = last_layer
        self.eps = eps
        self._model_db = model_db
        self.loss_fn = loss

        self.model, self.model_params, self.opt, self.opt_params = self._int_model()

    def _int_model(self):
        model = Net(
            n_in=self.n_in,
            n_out=self.n_out,
            hidden_dims=self.hidden_dims,
            activ=self.activ,
            last_layer=self.last_layer,
        )
        params = get_parameters(model)
        # print(len(params), type(params))
        opt = Adam(params)
        opt_params = {"m": opt.m, "v": opt.v}

        model_params = get_no_grad_params(model)
        return model, model_params, opt, opt_params

    def predict(self, x, model_id):
        x = Tensor([x], requires_grad=True).reshape((1, self.n_in))
        _model_params = self.get_init_model(model_id=model_id)
        if _model_params is not None:
            actor_params, optim_params = _model_params
            actor_model = load_state_dict(self.model, actor_params)
        else:
            actor_model = self.model

        result = actor_model(x).data[0]
        return result

    def learn(self, x, y, model_id, print_cost=False):
        _model_params = self.get_init_model(model_id=model_id, mode="learn")
        model, model_params, opt, opt_params = self._int_model()
        actor_model = model

        if _model_params is not None:
            actor_params, opt_params, model_update_cnt = _model_params
            actor_model = load_state_dict(actor_model, actor_params)
            actor_params_with_grad = get_parameters(actor_model)
            opt = Adam(actor_params_with_grad)
            opt.m = opt_params["m"]
            opt.v = opt_params["v"]
            opt.t = model_update_cnt

        x_obs = Tensor([x], requires_grad=True).reshape((1, self.n_in))
        y_obs = Tensor([y], requires_grad=True).reshape((1, self.n_out))
        output = actor_model(x_obs)
        if self.loss_fn == "MSE":
            loss = nn.MSELoss()(output, y_obs)
        elif self.loss_fn == "BCE":
            loss = nn.BCELoss()(output, y_obs)
        else:
            raise ValueError(f"loss {self.loss_fn} is not implemented.")

        opt.zero_grad()
        loss.backward()
        opt_params_updated = opt.step()
        opt_params["m"] = copy.deepcopy(opt.m)
        opt_params["v"] = copy.deepcopy(opt.v)
        _actor_params_new = get_state_dict(actor_model)
        actor_params_new = {}
        i = 0
        for model_key in list(_actor_params_new.keys()):
            actor_params_new[model_key] = opt_params_updated[i]  # .data
            i += 1

        new_model_params = {}
        new_model_params["params"] = actor_params_new
        new_model_params["optim"] = opt_params
        self.save_model(new_model_params, model_id=model_id)

        if print_cost:
            print(loss.data)
            return loss

    def get_init_model(self, model_id, mode="predict"):
        _model = self._model_db.get(model_id)
        if _model is not None:
            actor_params = _model["params"]
            optim_params = _model["optim"]
            model_update_cnt = self._model_db.get(f"{model_id}:updatecnt", 0)
            if mode == "learn":
                return actor_params, optim_params, model_update_cnt
            return actor_params, optim_params

        return _model

    def save_model(self, model, model_id):
        self._model_db.incr(f"{model_id}:updatecnt")
        self._model_db.set(model_id, model)
