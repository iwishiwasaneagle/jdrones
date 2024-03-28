#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from typing import Optional
from typing import Tuple

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device


class DenseNetModule(th.nn.Module):
    def __init__(self, in_dim, activation_fn):
        super().__init__()
        self.linear = th.nn.Linear(in_dim, in_dim)
        self.activation = activation_fn()

    def forward(self, feature):
        x = self.linear(feature)
        x = self.activation(x)
        return th.cat((x, feature), 1)


class DenseNetExtractor(th.nn.Module):

    def __init__(
        self,
        feature_dim: int,
        net_arch: int,
        activation_fn,
        device="auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net = []
        value_net = []

        # Iterate through the policy layers and build the policy net
        for n in range(net_arch):
            d = feature_dim * (2**n)
            policy_net.append(DenseNetModule(d, activation_fn))
            value_net.append(DenseNetModule(d, activation_fn))

        self.latent_dim_vf = self.latent_dim_pi = 2 * d

        self.policy_net = th.nn.Sequential(*policy_net).to(device)
        self.value_net = th.nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ActorCriticDenseNetPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        net_arch: Optional[int] = None,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = 2
        super().__init__(*args, net_arch=net_arch, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DenseNetExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
