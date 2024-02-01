from tensordict.nn import TensorDictModule
from torch import nn, Tensor

from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    StepCounter,
    TransformedEnv,
    SerialEnv,
    ToTensorImage,
    ParallelEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import OneHotCategorical, ProbabilisticActor

max_step = 200
n_env = 4
env_id = "CartPole-v1"
device = "cuda:0"


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
    env = ParallelEnv(n_env, EnvCreator(lambda: build_cpu_single_env()), device=device)
    policy_module = build_actor(env)
    policy_module.to(device)
    policy_module(env.reset())
    collector = SyncDataCollector(
        env,
        policy=policy_module,
        frames_per_batch=(max_step + 3) * 4,
        total_frames=10 * (max_step + 3) * 4,
        reset_at_each_iter=True,
        device=device,
    )
    # for i in range(10):
    for batches in collector:
        batches = env.rollout((max_step + 3), policy=policy_module, break_when_any_done=False)
        # batches = collector.next()
        max_step_count = batches["next", "step_count"].max().item()
        if max_step_count > max_step:
            print("Problem!")
            print(max_step_count)
            break
    else:
        print("No problem!")


class PixelModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.LazyLinear(n)

    def forward(self, inputs):
        out = self.w(inputs.reshape(*inputs.shape[:-3], -1))
        return out
