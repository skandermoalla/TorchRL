import torch.utils.benchmark
from torchrl.envs import GymEnv, ParallelEnv, SerialEnv

if __name__ == "__main__":
    for mp_start_method in [None, "spawn", "fork"]:
        for worker_device in ["cpu"]:
            for main_device in ["mps", "cpu"]:
                print(mp_start_method, worker_device, main_device)

                env = ParallelEnv(
                    16,
                    lambda: GymEnv("CartPole-v1", device=worker_device),
                    device=torch.device(main_device),
                    non_blocking=True,
                    mp_start_method=mp_start_method,
                )

                env.rollout(2)
                print(
                    torch.utils.benchmark.Timer(
                        "env.rollout(1000, break_when_any_done=False)", globals=globals()
                    ).adaptive_autorange()
                )

                torch.manual_seed(0)
                env.set_seed(0)
                rollout = env.rollout(1000, break_when_any_done=False)
                act0 = rollout["observation"].squeeze()

                torch.manual_seed(0)
                env.set_seed(0)
                rollout = env.rollout(1000, break_when_any_done=False)
                act1 = rollout["observation"].squeeze()

                torch.testing.assert_close(act0, act1)
                env.close()
                del env
