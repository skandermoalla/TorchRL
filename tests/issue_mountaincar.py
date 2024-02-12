from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    ExplorationType,
    ParallelEnv,
    SerialEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import OneHotCategorical, ProbabilisticActor

max_step = 210
n_env = 4
env_id = "MountainCar-v0"
NATIVE_TRUNCATION = 200
max_step = min(max_step, NATIVE_TRUNCATION)
device = "mps"


def build_cpu_single_env():
    env = GymEnv(env_id, device="cpu")
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
    env = SerialEnv(n_env, lambda: build_cpu_single_env(), device=device)
    # env = TransformedEnv(env)
    policy_module = build_actor(env)
    policy_module.to(device)
    policy_module(env.reset())

    collector = SyncDataCollector(
        env,
        policy=policy_module,
        frames_per_batch=(max_step + 3) * 4,
        total_frames=10 * (max_step + 3) * 4,
        reset_at_each_iter=False,
        device=device,
    )
    max_step_count = 200
    for batches in collector:
        max_step_count = batches["next", "step_count"].max().item()
        if max_step_count > max_step:
            print(max_step_count)
            print(batches["next", "step_count"][:, -5:])
            print("Problem! Got higher than max step count.")
            break
        elif max_step_count < max_step:
            print(max_step_count)
            print(batches["next", "step_count"][:, -5:])
            print("Problem: Got less than max step count!")
            break
    else:
        print(batches["next", "step_count"][:, -5:])
        print("No problem!")
