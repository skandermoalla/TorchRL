from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType
from torchrl.collectors import SyncDataCollector
from torchrl.envs import RewardSum, StepCounter, TransformedEnv, ParallelEnv, EnvCreator
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, OneHotCategorical

N = 202
env_id = "MountainCar-v0"
device = "cuda:0"


def build_single_env():
    env = GymEnv(env_id)
    env = TransformedEnv(env)
    env.append_transform(StepCounter(max_steps=N, truncated_key="truncated_sc"))
    return env


if __name__ == "__main__":
    env = ParallelEnv(4, EnvCreator(lambda: build_single_env()), device=device)
    # env = TransformedEnv(env)
    # Comment the line above and the problem goes away.

    policy_module = ProbabilisticActor(
        module=TensorDictModule(
            nn.LazyLinear(env.action_spec.space.n, device=device),
            in_keys=["observation"],
            out_keys=["logits"],
        ),
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        in_keys=["logits"],  # in_keys of the distribution.
        default_interaction_type=ExplorationType.RANDOM,
    )
    policy_module(env.reset())

    collector = SyncDataCollector(
        env,
        policy=policy_module,
        frames_per_batch=N * 4,
        total_frames=100 * N * 4,
        reset_at_each_iter=True,
        device=device,
    )
    max_step_count = 200
    for batches in collector:
        max_step_count = batches["next", "step_count"].max().item()
        if max_step_count > 200:
            print("Problem!")
            print(max_step_count)
            break
