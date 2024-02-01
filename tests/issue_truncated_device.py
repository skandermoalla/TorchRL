from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType, SerialEnv
from torchrl.envs import StepCounter, TransformedEnv, ParallelEnv, EnvCreator
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, OneHotCategorical

max_step = 200
env_id = "CartPole-v1"
device = "cuda:0"


def build_single_env():
    env = GymEnv(env_id)
    env = TransformedEnv(env)
    env.append_transform(StepCounter(max_steps=max_step))
    return env


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
    env = ParallelEnv(4, EnvCreator(lambda: build_single_env()), device=device)
    env = TransformedEnv(env)
    # Comment the line above to remove the problem.

    policy_module = build_actor(env)
    policy_module.to(device)
    policy_module(env.reset())
    for i in range(10):
        batches = env.rollout(max_step + 3, policy=policy_module, break_when_any_done=False)
