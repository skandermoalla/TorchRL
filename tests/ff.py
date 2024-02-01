import torch
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv


def run():
    env2 = ParallelEnv(3, EnvCreator(lambda: GymEnv("CartPole-v1")))
    print(torch.get_num_threads())

    env2.reset()
    print(torch.get_num_threads())

    env1 = ParallelEnv(8, EnvCreator(lambda: GymEnv("CartPole-v1")))
    print(torch.get_num_threads())

    env1.reset()
    print(torch.get_num_threads())

    env1.reset()
    print(torch.get_num_threads())


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(torch.get_num_threads())
    run()
