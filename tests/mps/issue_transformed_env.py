from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import OneHotCategorical, ProbabilisticActor

max_step = 10
env_id = "CartPole-v1"
device = "mps"


def build_actor(env):
    return ProbabilisticActor(
        module=TensorDictModule(
            nn.LazyLinear(env.action_spec.space.n),
            in_keys=["observation"],
            out_keys=["logits"],
        ),
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        in_keys=["logits"],
        default_interaction_type=ExplorationType.RANDOM,
    )


if __name__ == "__main__":
    env = ParallelEnv(4, lambda: GymEnv(env_id), device=device)
    # Changing to serial env removes the problem.
    env = TransformedEnv(env)
    # Or removing the transformed env removes the problem.

    policy_module = build_actor(env)
    policy_module.to(device)
    policy_module(env.reset())
    for i in range(10):
        batches = env.rollout(max_step + 3, policy=policy_module, break_when_any_done=False)
