from tensordict.nn import TensorDictModule
from torch import nn

from tests.custom_step_counter import CustomStepCounter
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    StepCounter,
    TransformedEnv,
    ParallelEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import OneHotCategorical, ProbabilisticActor

max_step = 10
n_env = 4
env_id = "MountainCar-v0"
device = "cuda:0"


def build_cpu_single_env():
    env = GymEnv(env_id, device="cpu")
    env = TransformedEnv(env)
    env.append_transform(
        CustomStepCounter(max_steps=max_step, truncated_key="terminated", set_terminated=True)
    )
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
    env = ParallelEnv(n_env, lambda: build_cpu_single_env(), device=device)
    env = TransformedEnv(env)
    policy_module = build_actor(env)
    policy_module.to(device)
    policy_module(env.reset())
    for i in range(10):
        batches = env.rollout((max_step + 3), policy=policy_module, break_when_any_done=False)
        max_step_count = batches["next", "step_count"].max().item()
        # print(max_step_count)
        # print(batches["next", "step_count"])
        if max_step_count > max_step:
            print("Problem 1!")
            print(max_step_count)
            print(batches["next", "step_count"])
            break
        elif max_step_count < max_step:
            print("Problem 2!")
            print(max_step_count)
            print(batches["next", "step_count"])
            break
    else:
        print("No problem!")
        print(batches["next", "terminated"])
        print(batches["next", "truncated"])
