from tinynet import Tensor, draw_dot
import tinynet.nn as nn
from tinynet.optim import SGD, Adam
from tinynet.state import (
    get_parameters,
    load_state_dict,
    get_no_grad_params,
    get_state_dict,
)

import numpy as np
import random
import copy


class DDQNNet:
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
        x = self.l3(x)
        return x


class DDQNAgent:
    def __init__(
        self,
        n_in,
        n_out,
        hidden_dims=[32, 16],
        activ="tanh",
        last_layer="iden",
        model_db=None,
        eps=0.1,
        update_cnt=5,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_dims = hidden_dims
        self.activ = activ
        self.last_layer = last_layer
        self.eps = eps
        self._model_db = model_db
        self.update_cnt = update_cnt
        self.actions = [a for a in range(n_out)]

        (
            self.actor_model,
            self.actor_params,
            self.opt,
            self.opt_params,
        ) = self._int_model()
        self.critic_model, self.critic_params, _, _ = self._int_model()
        self.loss_fn = nn.MSELoss()

    def _int_model(self):
        model = DDQNNet(
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

    def act(self, x, model_id, allowed=None, not_allowed=None):
        if allowed is None:
            valid_actions = self.actions
        else:
            valid_actions = allowed
        if not_allowed is not None:
            valid_actions = self._get_valid_actions(forbidden_actions=not_allowed)

        if random.random() < self.eps:
            action = random.choice(valid_actions)
        else:
            x = Tensor([x], requires_grad=True).reshape((1, self.n_in))
            _model_params = self.get_init_model(model_id=model_id)
            if _model_params is not None:
                actor_params, critic_params, optim_params = _model_params
                actor_model = load_state_dict(self.actor_model, actor_params)
            else:
                actor_model = self.actor_model
            if self.last_layer == "sigmoid":
                action_probs = actor_model(x=x).sigmoid().data[0]
            else:
                action_probs = actor_model(x=x).data[0]
            # print(action_probs, "action_probs")
            action_probs_dict = {a: action_probs[a] for a in valid_actions}
            action = self.argmax_rand(action_probs_dict)

        return action

    def learn(
        self,
        state,
        next_state,
        action,
        reward,
        model_id,
        done=False,
        print_cost=False,
    ):
        _model_params = self.get_init_model(model_id=model_id, mode="learn")
        model, model_params, opt, opt_params = self._int_model()
        actor_model = model
        critic_model = actor_model

        if _model_params is not None:
            actor_params, critic_params, opt_params, model_update_cnt = _model_params
            actor_model = load_state_dict(actor_model, actor_params)
            critic_model = load_state_dict(critic_model, critic_params)
            actor_params_with_grad = get_parameters(actor_model)
            opt = Adam(actor_params_with_grad)
            opt.m = opt_params["m"]
            opt.v = opt_params["v"]
            opt.t = model_update_cnt

        state_obs = Tensor([state], requires_grad=True).reshape((1, self.n_in))
        next_state_obs = Tensor([next_state], requires_grad=True).reshape(
            (1, self.n_in)
        )
        if self.last_layer == "sigmoid":
            value_next = critic_model(next_state_obs).sigmoid()
        else:
            value_next = critic_model(next_state_obs)
        a = np.argmax(value_next.data[0], axis=0)
        if self.last_layer == "sigmoid":
            Q_old = actor_model(state_obs).sigmoid()
        else:
            Q_old = actor_model(state_obs)
        Q = copy.deepcopy(Q_old.data[0])
        Q[action] = reward + (1 - np.logical_not(done)) * 0.99 * value_next.data[0][a]
        Q_tensor = Tensor([Q], requires_grad=True).reshape((1, self.n_out))
        loss_val = self.loss_fn(Q_old, Q_tensor)
        opt.zero_grad()
        loss_val.backward()
        opt_params_updated = opt.step()
        opt_params["m"] = copy.deepcopy(opt.m)
        opt_params["v"] = copy.deepcopy(opt.v)
        # opt_params_updated=opt.params

        _actor_params_new = get_state_dict(actor_model)
        actor_params_new = {}
        i = 0
        for model_key in list(_actor_params_new.keys()):
            actor_params_new[model_key] = opt_params_updated[i]  # .data
            i += 1
        # print(actor_params_new,"actor_params_new")

        critic_params_new = get_no_grad_params(critic_model)

        new_model_params = {}
        new_model_params["actor"] = {}
        new_model_params["critic"] = {}
        new_model_params["actor"]["params"] = actor_params_new
        new_model_params["critic"]["params"] = critic_params_new
        new_model_params["actor"]["optim"] = opt_params
        self.save_model(new_model_params, model_id=model_id)

        if print_cost:
            print(loss_val.data)

    def get_init_model(self, model_id, mode="predict"):
        _model = self._model_db.get(model_id)
        if _model is not None:
            actor_params = _model["actor"]["params"]
            critic_params = _model["critic"]["params"]
            model_update_cnt = self._model_db.get(f"{model_id}:updatecnt", 0)
            if model_update_cnt % self.update_cnt == 0:
                critic_params = copy.deepcopy(actor_params)
            optim_params = _model["actor"]["optim"]
            if mode == "learn":
                return actor_params, critic_params, optim_params, model_update_cnt
            return actor_params, critic_params, optim_params

        return _model

    def save_model(self, model, model_id):
        self._model_db.incr(f"{model_id}:updatecnt")
        self._model_db.set(model_id, model)

    def _get_valid_actions(self, forbidden_actions, all_actions=None):
        """
        Given a set of forbidden action IDs, return a set of valid action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[Set[ActionId]]
            The set of forbidden action IDs.

        Returns
        -------
        valid_actions: Set[ActionId]
            The list of valid (i.e. not forbidden) action IDs.
        """
        if all_actions is None:
            all_actions = self.actions
        if forbidden_actions is None:
            forbidden_actions = set()
        else:
            forbidden_actions = set(forbidden_actions)

        if not all(a in all_actions for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = set(all_actions) - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError(
                "All actions are forbidden. You must allow at least 1 action."
            )

        valid_actions = list(valid_actions)
        return valid_actions

    def argmax_rand(self, dict_arr):
        """Return key with maximum value, break ties randomly."""
        assert isinstance(dict_arr, dict)
        # Find the maximum value in the dictionary
        max_value = max(dict_arr.values())
        # Get a list of keys with the maximum value
        max_keys = [key for key, value in dict_arr.items() if value == max_value]
        # Randomly select one key from the list
        selected_key = random.choice(max_keys)
        # Return the selected key
        return selected_key
