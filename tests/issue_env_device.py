from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import OneHotCategorical, ProbabilisticActor

N = 202
env_id = "MountainCar-v0"
device = "mps"  # "cuda:0


def build_single_env():
    env = GymEnv(env_id)
    env = TransformedEnv(env)
    env.append_transform(StepCounter(max_steps=N, truncated_key="truncated_sc"))
    return env


if __name__ == "__main__":
    env = ParallelEnv(4, EnvCreator(lambda: build_single_env()), device=device)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())

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

    collector = SyncDataCollector(
        env,
        policy=policy_module,
        frames_per_batch=N * 4,
        total_frames=100 * N * 4,
        reset_at_each_iter=True,
    )
    max_step_count = 200
    for batches in collector:
        best_return = batches["next", "episode_reward"][batches["next", "done"]].max().item()
        max_step_count = batches["next", "step_count"].max().item()
        if max_step_count > 200:
            print("Problem!")
            print(best_return, max_step_count)
            break

    print(f'done: {batches["next", "done"].squeeze(-1).nonzero().squeeze(-1)}')
    print(f'truncated: {batches["next", "truncated"].squeeze(-1).nonzero().squeeze(-1)}')
    print(f'truncated_sc: {batches["next", "truncated_sc"].squeeze(-1).nonzero().squeeze(-1)}')
    print(f'terminated: {batches["next", "terminated"].squeeze(-1).nonzero().squeeze(-1)}')
    pass
